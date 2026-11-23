/** The connected process owns its runtime version — and the commit it actually LOADED — not a
 *  checkout on disk. `displayVersion` carries the distance past the release tag (`0.21.5+1913`);
 *  a backend that predates it only reports the bare release. `commit` is the commit behind the
 *  running process, which on a remote connection is a different tree from the client's install
 *  stamp (the About panel used to print that stamp under a "backend" verdict). */
export interface GatewayBuild {
  version: string
  commit: string
}

export async function resolveGatewayBuild(request: (path: string) => Promise<unknown>): Promise<GatewayBuild> {
  try {
    const status: unknown = await request('/api/health')

    if (status && typeof status === 'object') {
      const version =
        'displayVersion' in status && typeof status.displayVersion === 'string' && status.displayVersion
          ? status.displayVersion
          : 'version' in status && typeof status.version === 'string'
            ? status.version
            : ''

      const commit = 'commit' in status && typeof status.commit === 'string' ? status.commit.trim() : ''

      return { version, commit }
    }
  } catch {
    // An offline gateway has no known running version.
  }

  return { version: '', commit: '' }
}

export async function resolveGatewayVersion(request: (path: string) => Promise<unknown>): Promise<string> {
  return (await resolveGatewayBuild(request)).version
}
