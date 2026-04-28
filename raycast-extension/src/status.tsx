import { List, Icon, showToast, Toast } from "@raycast/api";
import { useState, useEffect } from "react";
import { getStatus, StatusResponse, formatBytes } from "./api";

export default function StatusCommand() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchStatus() {
      try {
        const data = await getStatus();
        setStatus(data);
      } catch (error) {
        showToast({
          style: Toast.Style.Failure,
          title: "Failed to connect",
          message: "Is the SemFile server running? Run: semfile serve",
        });
      } finally {
        setIsLoading(false);
      }
    }
    fetchStatus();
  }, []);

  return (
    <List isLoading={isLoading}>
      {status && (
        <>
          <List.Section title="Index">
            <List.Item icon={Icon.Document} title="Total Files" accessories={[{ text: String(status.total) }]} />
            {Object.entries(status.by_type)
              .sort(([, a], [, b]) => b - a)
              .map(([type, count]) => (
                <List.Item key={type} icon={Icon.Dot} title={`  ${type}`} accessories={[{ text: String(count) }]} />
              ))}
          </List.Section>
          <List.Section title="Storage">
            <List.Item icon={Icon.HardDrive} title="Total" accessories={[{ text: formatBytes(status.storage.total_bytes) }]} />
            <List.Item icon={Icon.Dot} title="  Database" accessories={[{ text: formatBytes(status.storage.db_bytes) }]} />
            <List.Item icon={Icon.Dot} title="  Thumbnails" accessories={[{ text: formatBytes(status.storage.thumbnail_bytes) }]} />
          </List.Section>
          <List.Section title="Config">
            <List.Item icon={Icon.Gear} title="Embedding Dimensions" accessories={[{ text: String(status.embedding_dimensions) }]} />
          </List.Section>
        </>
      )}
    </List>
  );
}
