import {
  ActionPanel,
  Action,
  List,
  Form,
  Icon,
  Color,
  showToast,
  Toast,
  useNavigation,
} from "@raycast/api";
import { useState, useEffect, useRef } from "react";
import { searchFiles, SearchResultItem, getThumbnailUrl, formatBytes } from "./api";

const FILE_TYPE_ICONS: Record<string, { icon: Icon; color: Color }> = {
  image: { icon: Icon.Image, color: Color.Green },
  video: { icon: Icon.Video, color: Color.Blue },
  audio: { icon: Icon.Music, color: Color.Purple },
  document: { icon: Icon.Document, color: Color.Orange },
  text: { icon: Icon.Text, color: Color.SecondaryText },
};

const RERANK_TOP_N = 20;

function basename(path: string): string {
  const trimmed = path.replace(/\/+$/, "");
  const idx = trimmed.lastIndexOf("/");
  return idx >= 0 ? trimmed.slice(idx + 1) : trimmed;
}

function dirSummary(dirs: string[]): string {
  if (dirs.length === 0) return "";
  if (dirs.length === 1) return `in ${basename(dirs[0])}`;
  return `in ${dirs.length} folders`;
}

export default function SearchCommand() {
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [fileTypeFilter, setFileTypeFilter] = useState("all");
  const [rerank, setRerank] = useState(false);
  const [directories, setDirectories] = useState<string[]>([]);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { push } = useNavigation();

  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    if (!searchText.trim()) {
      setResults([]);
      return;
    }

    const delay = rerank ? 600 : 300;
    debounceRef.current = setTimeout(async () => {
      setIsLoading(true);
      try {
        const response = await searchFiles(searchText, {
          type: fileTypeFilter,
          limit: 30,
          rerank,
          rerankTopN: RERANK_TOP_N,
          dirs: directories,
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
    }, delay);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [searchText, fileTypeFilter, rerank, directories]);

  const toggleRerank = () => {
    const next = !rerank;
    setRerank(next);
    showToast({
      style: Toast.Style.Success,
      title: next ? "Rerank on" : "Rerank off",
      message: next ? "Multimodal Gemini rerank applied to top candidates" : "Using vector similarity only",
    });
  };

  const openDirFilter = () => {
    push(
      <DirFilterForm
        current={directories}
        onSubmit={(paths) => {
          setDirectories(paths);
          showToast({
            style: Toast.Style.Success,
            title: paths.length === 0 ? "Folder filter cleared" : `Searching ${dirSummary(paths)}`,
          });
        }}
      />,
    );
  };

  const clearDirs = () => {
    setDirectories([]);
    showToast({ style: Toast.Style.Success, title: "Folder filter cleared" });
  };

  const placeholderParts = ["Describe what you're looking for..."];
  if (rerank) placeholderParts.push("Rerank on");
  if (directories.length > 0) placeholderParts.push(dirSummary(directories));
  const placeholder = placeholderParts.length === 1
    ? placeholderParts[0]
    : `${placeholderParts[0]} (${placeholderParts.slice(1).join(", ")})`;

  const globalActions = (
    <ActionPanel.Section>
      <Action
        title={rerank ? "Disable Rerank" : "Enable Rerank"}
        icon={rerank ? Icon.StarDisabled : Icon.Stars}
        shortcut={{ modifiers: ["cmd"], key: "r" }}
        onAction={toggleRerank}
      />
      <Action
        title="Filter by Folder…"
        icon={Icon.Folder}
        shortcut={{ modifiers: ["cmd"], key: "d" }}
        onAction={openDirFilter}
      />
      {directories.length > 0 && (
        <Action
          title="Clear Folder Filter"
          icon={Icon.XMarkCircle}
          shortcut={{ modifiers: ["cmd", "shift"], key: "d" }}
          onAction={clearDirs}
        />
      )}
    </ActionPanel.Section>
  );

  return (
    <List
      isLoading={isLoading}
      onSearchTextChange={setSearchText}
      searchBarPlaceholder={placeholder}
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
        <List.EmptyView
          icon={Icon.MagnifyingGlass}
          title="Type to search"
          description={
            directories.length > 0
              ? `Filtered to ${dirSummary(directories)} (Cmd+Shift+D to clear)`
              : "Cmd+R toggles rerank · Cmd+D filters by folder"
          }
          actions={<ActionPanel>{globalActions}</ActionPanel>}
        />
      ) : results.length === 0 && !isLoading ? (
        <List.EmptyView
          icon={Icon.XMarkCircle}
          title="No results"
          description="Try a different search query"
          actions={<ActionPanel>{globalActions}</ActionPanel>}
        />
      ) : (
        results.map((item) => {
          const typeInfo = FILE_TYPE_ICONS[item.file_type] || { icon: Icon.Document, color: Color.SecondaryText };
          const thumbnailUrl = getThumbnailUrl(item.thumbnail_url);
          const subtitle =
            item.rerank_score !== null && item.rerank_score !== undefined
              ? `★ ${item.rerank_score.toFixed(1)} · ${(item.similarity * 100).toFixed(0)}%`
              : `${(item.similarity * 100).toFixed(0)}%`;

          const accessories: List.Item.Accessory[] = [];
          if (rerank) {
            accessories.push({ icon: { source: Icon.Stars, tintColor: Color.Yellow }, tooltip: "Reranked" });
          }
          accessories.push({ tag: { value: item.file_type, color: typeInfo.color } });

          return (
            <List.Item
              key={item.file_path}
              icon={{ source: typeInfo.icon, tintColor: typeInfo.color }}
              title={item.filename}
              subtitle={subtitle}
              accessories={accessories}
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
                      {item.rerank_score !== null && item.rerank_score !== undefined && (
                        <List.Item.Detail.Metadata.Label
                          title="Rerank Score"
                          text={`${item.rerank_score.toFixed(2)} / 10`}
                        />
                      )}
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
                  {globalActions}
                </ActionPanel>
              }
            />
          );
        })
      )}
    </List>
  );
}

function DirFilterForm({
  current,
  onSubmit,
}: {
  current: string[];
  onSubmit: (paths: string[]) => void;
}) {
  const { pop } = useNavigation();
  return (
    <Form
      navigationTitle="Filter by Folder"
      actions={
        <ActionPanel>
          <Action.SubmitForm
            title="Apply Filter"
            icon={Icon.Check}
            onSubmit={(values: { folders: string[] }) => {
              onSubmit(values.folders ?? []);
              pop();
            }}
          />
          <Action
            title="Clear Filter"
            icon={Icon.XMarkCircle}
            shortcut={{ modifiers: ["cmd", "shift"], key: "d" }}
            onAction={() => {
              onSubmit([]);
              pop();
            }}
          />
        </ActionPanel>
      }
    >
      <Form.Description text="Pick one or more folders. Search will only return files under these paths. Leave empty to search everything indexed." />
      <Form.FilePicker
        id="folders"
        title="Folders"
        allowMultipleSelection
        canChooseDirectories
        canChooseFiles={false}
        defaultValue={current}
      />
    </Form>
  );
}
