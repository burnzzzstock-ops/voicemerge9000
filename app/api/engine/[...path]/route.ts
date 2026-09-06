import { env } from 'cloudflare:workers';

type EngineBindings = { RVC_ENGINE_URL?: string; RVC_ENGINE_API_TOKEN?: string };

function engineSettings() {
  const bindings = env as unknown as EngineBindings;
  return {
    origin: (bindings.RVC_ENGINE_URL || process.env.RVC_ENGINE_URL)?.replace(/\/$/, ''),
    token: bindings.RVC_ENGINE_API_TOKEN || process.env.RVC_ENGINE_API_TOKEN,
  };
}

function jsonError(message: string, status: number) {
  return Response.json({ error: message }, { status });
}

async function proxy(request: Request, context: { params: Promise<{ path: string[] }> }) {
  const { origin: engineOrigin, token: engineToken } = engineSettings();
  if (!engineOrigin) {
    return jsonError('The integrated RVC engine has not been connected to this deployment.', 503);
  }

  let origin: URL;
  try {
    origin = new URL(engineOrigin);
  } catch {
    return jsonError('The configured RVC engine URL is invalid.', 503);
  }
  if (!['https:', 'http:'].includes(origin.protocol)) {
    return jsonError('The configured RVC engine URL must use HTTP or HTTPS.', 503);
  }

  const { path } = await context.params;
  const safePath = path.filter((part) => part !== '.' && part !== '..' && /^[A-Za-z0-9._-]+$/.test(part));
  if (safePath.length !== path.length) return jsonError('Invalid engine path.', 400);
  const target = new URL(`/api/v1/${safePath.join('/')}${new URL(request.url).search}`, origin);
  const headers = new Headers();
  for (const name of ['accept', 'content-type', 'content-length', 'range']) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  if (engineToken) headers.set('authorization', `Bearer ${engineToken}`);

  try {
    const upstream = await fetch(target, {
      method: request.method,
      headers,
      body: ['GET', 'HEAD'].includes(request.method) ? undefined : request.body,
      redirect: 'manual',
      signal: request.signal,
    });
    const responseHeaders = new Headers();
    for (const name of ['accept-ranges', 'content-disposition', 'content-length', 'content-range', 'content-type']) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }
    responseHeaders.set('cache-control', 'no-store');
    return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
  } catch {
    return jsonError('The RVC engine could not be reached.', 502);
  }
}

export const GET = proxy;
export const HEAD = proxy;
export const POST = proxy;
export const DELETE = proxy;
