import type { NextRequest } from "next/server";

async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  const base = process.env.LANTERN_BACKEND_URL || "http://127.0.0.1:8000";
  try {
    const target = new URL(
      base.replace(/\/$/, "") +
        "/api/" +
        path.map(encodeURIComponent).join("/"),
    );
    target.search = request.nextUrl.search;
    const payload =
      request.method === "GET" || request.method === "HEAD"
        ? undefined
        : await request.arrayBuffer();
    const upstream = await fetch(target, {
      method: request.method,
      headers: {
        "Content-Type":
          request.headers.get("content-type") || "application/json",
      },
      body: payload?.byteLength ? payload : undefined,
      cache: "no-store",
      signal: request.signal,
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type":
          upstream.headers.get("content-type") || "application/json",
        "Cache-Control": "no-store",
        ...(upstream.headers.get("content-disposition")
          ? {
              "Content-Disposition": upstream.headers.get(
                "content-disposition",
              )!,
            }
          : {}),
      },
    });
  } catch {
    return Response.json(
      {
        detail:
          "Cannot reach the Python backend. Start it and check LANTERN_BACKEND_URL.",
      },
      { status: 502 },
    );
  }
}

export { proxy as GET, proxy as POST, proxy as PATCH, proxy as DELETE };
