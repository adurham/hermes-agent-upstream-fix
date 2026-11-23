import assert from 'node:assert/strict'

import { test } from 'vitest'

import { resolveGatewayBuild, resolveGatewayVersion } from './gateway-version'

test('version follows each supplied gateway response without caching', async (): Promise<void> => {
  let version: string = '1.2.3'

  const request: (endpoint: string) => Promise<{ version: string }> = async (
    endpoint: string
  ): Promise<{ version: string }> => {
    assert.equal(endpoint, '/api/health')

    return { version }
  }

  assert.equal(await resolveGatewayVersion(request), '1.2.3')
  version = '4.5.6'
  assert.equal(await resolveGatewayVersion(request), '4.5.6')
})

test('unavailable gateway version stays unknown instead of using another install', async () => {
  for (const response of [null, {}, { version: 42 }, { version: '' }]) {
    assert.equal(await resolveGatewayVersion(async () => response), '')
  }

  assert.equal(
    await resolveGatewayVersion(async () => {
      throw new Error('offline')
    }),
    ''
  )
})

test('the distance past the release tag wins over the bare release', async (): Promise<void> => {
  const ahead: { version: string; displayVersion: string } = { version: '0.21.5', displayVersion: '0.21.5+1913' }
  assert.equal(await resolveGatewayVersion(async () => ahead), '0.21.5+1913')
  assert.equal(await resolveGatewayVersion(async () => ({ ...ahead, displayVersion: '' })), '0.21.5')
})

test('the build carries the commit the backend LOADED, not just its version', async (): Promise<void> => {
  const health = {
    version: '0.23.3',
    displayVersion: '0.23.3+63600',
    commit: ' 61c33461ce55df470aa76ae1d81898f45ea7dea0 '
  }

  assert.deepEqual(await resolveGatewayBuild(async () => health), {
    version: '0.23.3+63600',
    commit: '61c33461ce55df470aa76ae1d81898f45ea7dea0'
  })
})

test('a backend that answers without a commit still resolves its version', async (): Promise<void> => {
  assert.deepEqual(await resolveGatewayBuild(async () => ({ version: '0.21.5' })), { version: '0.21.5', commit: '' })
  assert.deepEqual(await resolveGatewayBuild(async () => ({ commit: 42 })), { version: '', commit: '' })
  assert.deepEqual(
    await resolveGatewayBuild(async () => {
      throw new Error('offline')
    }),
    { version: '', commit: '' }
  )
})
