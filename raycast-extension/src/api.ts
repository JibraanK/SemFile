import { getPreferenceValues } from "@raycast/api";

interface Preferences {
  serverUrl: string;
}

export interface SearchResultItem {
  file_path: string;
  filename: string;
  file_type: string;
  mime_type: string;
  file_size: number;
  similarity: number;
  rerank_score: number | null;
  thumbnail_url: string | null;
}

export interface SearchResponse {
  results: SearchResultItem[];
  query: string;
  count: number;
  reranked: boolean;
}

export interface StatusResponse {
  total: number;
  by_type: Record<string, number>;
  storage: {
    total_bytes: number;
    db_bytes: number;
    thumbnail_bytes: number;
  };
  embedding_dimensions: number;
}

function getBaseUrl(): string {
  const { serverUrl } = getPreferenceValues<Preferences>();
  return serverUrl || "http://localhost:19532";
}

export async function searchFiles(
  query: string,
  options?: {
    type?: string;
    limit?: number;
    rerank?: boolean;
    rerankTopN?: number;
    dirs?: string[];
  }
): Promise<SearchResponse> {
  const baseUrl = getBaseUrl();
  const params = new URLSearchParams({ q: query });
  if (options?.type && options.type !== "all") {
    params.append("type", options.type);
  }
  if (options?.limit) {
    params.set("limit", String(options.limit));
  }
  if (options?.rerank) {
    params.set("rerank", "true");
    if (options.rerankTopN) {
      params.set("rerank_top_n", String(options.rerankTopN));
    }
  }
  if (options?.dirs?.length) {
    for (const d of options.dirs) {
      params.append("dir", d);
    }
  }

  const response = await fetch(`${baseUrl}/search?${params}`);
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as SearchResponse;
}

export async function getStatus(): Promise<StatusResponse> {
  const baseUrl = getBaseUrl();
  const response = await fetch(`${baseUrl}/status`);
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as StatusResponse;
}

export interface IndexStatus {
  running: boolean;
  started_at: number | null;
  finished_at: number | null;
  path: string | null;
  file_types: string[] | null;
  stats: {
    scanned: number;
    indexed: number;
    skipped: number;
    failed: number;
    removed: number;
  } | null;
  error: string | null;
  count_at_start: number | null;
  count: number;
}

export interface TriggerIndexResponse {
  started: boolean;
  started_at: number;
}

export async function triggerIndex(options?: {
  path?: string;
  fileTypes?: string[];
}): Promise<TriggerIndexResponse> {
  const baseUrl = getBaseUrl();
  const body: Record<string, unknown> = {};
  if (options?.path) body.path = options.path;
  if (options?.fileTypes?.length) body.file_types = options.fileTypes;

  const response = await fetch(`${baseUrl}/index`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (response.status === 409) {
    const err = new Error("Index already running");
    (err as Error & { code?: number }).code = 409;
    throw err;
  }
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as TriggerIndexResponse;
}

export async function getIndexStatus(): Promise<IndexStatus> {
  const baseUrl = getBaseUrl();
  const response = await fetch(`${baseUrl}/index/status`);
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as IndexStatus;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { error?: string };
    if (body?.error) return body.error;
  } catch {
    // ignore non-JSON bodies
  }
  return `Server error: ${response.status}`;
}

export function getThumbnailUrl(thumbnailUrl: string | null): string | undefined {
  if (!thumbnailUrl) return undefined;
  const baseUrl = getBaseUrl();
  return `${baseUrl}${thumbnailUrl}`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  if (mb < 1024) return `${mb.toFixed(1)} MB`;
  const gb = mb / 1024;
  return `${gb.toFixed(1)} GB`;
}
