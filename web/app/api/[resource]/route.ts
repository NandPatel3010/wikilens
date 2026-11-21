import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

const allowed = new Set(["search", "benchmark", "health"]);

export async function GET(request: NextRequest, context: { params: Promise<{ resource: string }> }) {
  const { resource } = await context.params;
  if (!allowed.has(resource)) {
    return NextResponse.json({ detail: "Unknown resource." }, { status: 404 });
  }
  const base = process.env.SEARCH_API_URL;
  if (!base) {
    return NextResponse.json({ detail: "The search API URL has not been configured." }, { status: 503 });
  }
  const headers = new Headers();
  if (process.env.SEARCH_API_TOKEN) {
    headers.set("Authorization", "Bearer " + process.env.SEARCH_API_TOKEN);
  }
  try {
    const upstream = new URL(base.replace(/\/$/, "") + "/" + resource);
    for (const key of ["q", "mode", "page", "page_size", "allow_fallback"]) {
      const value = request.nextUrl.searchParams.get(key);
      if (resource === "search" && value !== null) upstream.searchParams.set(key, value);
    }
    const response = await fetch(upstream, {
      headers, cache: "no-store", redirect: "error", signal: AbortSignal.timeout(25_000),
    });
    if (response.status === 422) {
      return NextResponse.json(await response.json(), { status: 422 });
    }
    if (!response.ok) {
      return NextResponse.json({ detail: "The search service is temporarily unavailable." }, { status: 503 });
    }
    return NextResponse.json(await response.json());
  } catch {
    return NextResponse.json({ detail: "Could not reach the search service." }, { status: 503 });
  }
}
