import { useState } from 'react'

import { SubagentRow } from '@/app/agents'
import { ActivityTimerText } from '@/components/chat/activity-timer-text'
import { StatusRow } from '@/components/chat/status-row'
import { StatusSection } from '@/components/chat/status-section'
import { Codicon } from '@/components/ui/codicon'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { useViewedInterval } from '@/hooks/use-viewed-interval'
import { useI18n } from '@/i18n'
import { useSessionSlice } from '@/lib/use-session-slice'
import { $subagentsBySession, type SubagentProgress } from '@/store/subagents'

import { SubagentControls } from './subagent-controls'
import { SubagentTranscript } from './subagent-transcript'

interface SubagentSectionProps {
  sessionId: string
}

/** What the child is running ON and AS: effective model (⚠ when it silently
 *  failed over), persona and profile. All three answer "which subagent is this
 *  really", so they share one line ahead of the latest activity — the goal is
 *  the only thing that may be truncated away, never the identity. */
function identityParts(item: SubagentProgress): string[] {
  const model = item.model?.trim() || item.modelLabel?.trim() || ''

  return [
    model ? `${item.fallbackActive ? '⚠ ' : ''}${model}` : '',
    item.agentType?.trim() || '',
    item.profile?.trim() || ''
  ].filter(Boolean)
}

/** A composer-local roster: never borrow the global Agents panel's scope. */
export function SubagentSection({ sessionId }: SubagentSectionProps) {
  const { t } = useI18n()
  const items = useSessionSlice($subagentsBySession, sessionId)
  const live = items.filter(item => item.status === 'running' || item.status === 'queued')
  const [nowMs, setNowMs] = useState(Date.now)
  const [selected, setSelected] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const hasLive = live.length > 0

  useViewedInterval(() => setNowMs(Date.now()), 1000, hasLive)

  if (!hasLive) {
    return null
  }

  const row = (item: SubagentProgress) => {
    const identity = identityParts(item)
    const activity = item.stream.at(-1)?.text || (item.status === 'queued' ? t.agents.queued : t.agents.waitingActivity)

    return (
      <StatusRow
        expanded={selected === item.id}
        key={item.id}
        leading={
          <GlyphSpinner
            ariaLabel={item.status === 'queued' ? t.agents.queued : t.agents.running}
            className="text-(--ui-purple)"
            spinner="braille"
          />
        }
        onActivate={() => setSelected(selected === item.id ? null : item.id)}
        trailing={
          <ActivityTimerText
            className="shrink-0 text-[0.65rem]"
            seconds={Math.max(0, Math.floor((nowMs - item.startedAt) / 1000))}
          />
        }
        trailingVisible
      >
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs text-(--ui-text-primary)">{item.goal}</span>
          <span className="flex min-w-0 items-baseline gap-1.5 text-[0.68rem] leading-4">
            {identity.length > 0 && (
              <span
                className="max-w-[60%] shrink-0 truncate text-(--ui-text-secondary)"
                data-slot="composer-subagent-identity"
                title={item.modelLabel || undefined}
              >
                {identity.join(' · ')}
              </span>
            )}
            {identity.length > 0 && (
              <span aria-hidden className="shrink-0 text-(--ui-text-tertiary)">
                ·
              </span>
            )}
            <span className="min-w-0 truncate text-(--ui-text-tertiary)">{activity}</span>
          </span>
        </span>
      </StatusRow>
    )
  }

  const detail = live.find(item => item.id === selected)

  return (
    <div className="composer-no-drag min-w-0" data-slot="composer-subagents">
      <StatusSection
        collapsedIndicator={
          <GlyphSpinner
            ariaLabel={live.some(item => item.status === 'running') ? t.agents.running : t.agents.queued}
            className="text-(--ui-purple)"
            spinner="braille"
          />
        }
        icon={<Codicon className="text-(--ui-purple)" name="agent" size="0.8rem" />}
        label={t.statusStack.subagents(live.length)}
      >
        <div className="max-h-[25vh] overflow-y-auto overscroll-y-auto">{live.map(row)}</div>
        {detail && (
          <div
            className="status-subagent-detail max-h-[25vh] overflow-y-auto overscroll-y-auto pr-3 py-2"
            data-slot="composer-subagent-detail"
          >
            <SubagentControls
              key={`${sessionId}:${detail.id}`}
              sessionId={sessionId}
              setText={text => setDrafts(previous => ({ ...previous, [detail.id]: text }))}
              subagentId={detail.id}
              text={drafts[detail.id] ?? ''}
            />
            <SubagentRow node={{ ...detail, children: [] }} nowMs={nowMs} />
            <SubagentTranscript key={`tail:${sessionId}:${detail.id}`} sessionId={sessionId} subagentId={detail.id} />
          </div>
        )}
      </StatusSection>
    </div>
  )
}
