import { ActionPanel, Action, List, Icon, Color, showToast, Toast } from "@raycast/api";
import { useState, useEffect, useRef } from "react";
import { searchFiles, SearchResultItem, getThumbnailUrl, formatBytes } from "./api";

const FILE_TYPE_ICONS: Record<string, { icon: Icon; color: Color }> = {
  image: { icon: Icon.Image, color: Color.Green },
  video: { icon: Icon.Video, color: Color.Blue },
  audio: { icon: Icon.Music, color: Color.Purple },
  document: { icon: Icon.Document, color: Color.Orange },
  text: { icon: Icon.Text, color: Color.SecondaryText },
};

export default function SearchCommand() {
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [fileTypeFilter, setFileTypeFilter] = useState("all");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    if (!searchText.trim()) {
      setResults([]);
      return;
    }

    debounceRef.current = setTimeout(async () => {
      setIsLoading(true);
      try {
        const response = await searchFiles(searchText, {
          type: fileTypeFilter,
          limit: 30,
        });
        setResults(response.results);
      } catch (error) {
        showToast({
          style: Toast.Style.Failure,
          title: "Search failed",
          message: error instanceof Error ? error.message : "Is the SemFile server running? Run: semfile serve",
        });
        setResults([]);
      } finally {
        setIsLoading(false);
      }
    }, 300);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [searchText, fileTypeFilter]);

  return (
    <List
      isLoading={isLoading}
      onSearchTextChange={setSearchText}
      searchBarPlaceholder="Describe what you're looking for..."
      isShowingDetail
      throttle
      searchBarAccessory={
        <List.Dropdown tooltip="File Type" onChange={setFileTypeFilter}>
          <List.Dropdown.Item title="All Types" value="all" />
          <List.Dropdown.Item title="Images" value="image" icon={Icon.Image} />
          <List.Dropdown.Item title="Videos" value="video" icon={Icon.Video} />
          <List.Dropdown.Item title="Audio" value="audio" icon={Icon.Music} />
          <List.Dropdown.Item title="Documents" value="document" icon={Icon.Document} />
          <List.Dropdown.Item title="Text" value="text" icon={Icon.Text} />
        </List.Dropdown>
      }
    >
      {!searchText.trim() ? (
        <List.EmptyView icon={Icon.MagnifyingGlass} title="Type to search" description="Describe what you're looking for in natural language" />
      ) : results.length === 0 && !isLoading ? (
        <List.EmptyView icon={Icon.XMarkCircle} title="No results" description="Try a different search query" />
      ) : (
        results.map((item, index) => {
          const typeInfo = FILE_TYPE_ICONS[item.file_type] || { icon: Icon.Document, color: Color.SecondaryText };
          const thumbnailUrl = getThumbnailUrl(item.thumbnail_url);

          return (
            <List.Item
              key={item.file_path}
              icon={{ source: typeInfo.icon, tintColor: typeInfo.color }}
              title={item.filename}
              subtitle={`${(item.similarity * 100).toFixed(0)}%`}
              accessories={[{ tag: { value: item.file_type, color: typeInfo.color } }]}
              detail={
                <List.Item.Detail
                  markdown={
                    thumbnailUrl
                      ? `![thumbnail](${thumbnailUrl})`
                      : `*No preview available*`
                  }
                  metadata={
                    <List.Item.Detail.Metadata>
                      <List.Item.Detail.Metadata.Label title="Filename" text={item.filename} />
                      <List.Item.Detail.Metadata.Label title="Type" text={item.file_type} />
                      <List.Item.Detail.Metadata.Label title="Size" text={formatBytes(item.file_size)} />
                      <List.Item.Detail.Metadata.Label title="Similarity" text={`${(item.similarity * 100).toFixed(1)}%`} />
                      <List.Item.Detail.Metadata.Separator />
                      <List.Item.Detail.Metadata.Label title="Path" text={item.file_path} />
                    </List.Item.Detail.Metadata>
                  }
                />
              }
              actions={
                <ActionPanel>
                  <Action.Open title="Open File" target={item.file_path} />
                  <Action.ShowInFinder path={item.file_path} />
                  <Action.CopyToClipboard title="Copy Path" content={item.file_path} />
                  <Action.Open title="Open Containing Folder" target={item.file_path.substring(0, item.file_path.lastIndexOf("/"))} />
                </ActionPanel>
              }
            />
          );
        })
      )}
    </List>
  );
}
