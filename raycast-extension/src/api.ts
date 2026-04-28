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
  thumbnail_url: string | null;
}

export interface SearchResponse {
  results: SearchResultItem[];
  query: string;
  count: number;
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
  options?: { type?: string; limit?: number }
): Promise<SearchResponse> {
  const baseUrl = getBaseUrl();
  const params = new URLSearchParams({ q: query });
  if (options?.type && options.type !== "all") {
    params.append("type", options.type);
  }
  if (options?.limit) {
    params.set("limit", String(options.limit));
  }

  const response = await fetch(`${baseUrl}/search?${params}`);
  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }
  return (await response.json()) as SearchResponse;
}

export async function getStatus(): Promise<StatusResponse> {
  const baseUrl = getBaseUrl();
  const response = await fetch(`${baseUrl}/status`);
  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }
  return (await response.json()) as StatusResponse;
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
