import {
  ActionPanel,
  Action,
  Form,
  Detail,
  Icon,
  showToast,
  Toast,
  launchCommand,
  LaunchType,
  open,
} from "@raycast/api";
import { useEffect, useRef, useState } from "react";
import {
  triggerIndex,
  getIndexStatus,
  IndexStatus,
} from "./api";

const FILE_TYPES = ["image", "video", "audio", "document", "text"] as const;

type IndexParams = { path?: string; fileTypes?: string[] };

type View =
  | { kind: "form" }
  | { kind: "progress"; startWith?: IndexParams };

export default function IndexFilesCommand() {
  const [view, setView] = useState<View>({ kind: "form" });
  const [bootChecked, setBootChecked] = useState(false);
  const [bootError, setBootError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const status = await getIndexStatus();
        if (cancelled) return;
        if (status.running) {
          setView({ kind: "progress" });
        }
      } catch (error) {
        if (!cancelled) {
          setBootError(error instanceof Error ? error.message : "Unknown error");
          showToast({
            style: Toast.Style.Failure,
            title: "Cannot reach SemFile server",
            message: "Run: semfile serve",
          });
        }
      } finally {
        if (!cancelled) setBootChecked(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (view.kind === "progress") {
    return (
      <ProgressView
        startWith={view.startWith}
        onRunAnother={() => setView({ kind: "form" })}
      />
    );
  }

  return (
    <FormView
      isLoading={!bootChecked}
      bootError={bootError}
      onStart={(params) => {
        showToast({
          style: Toast.Style.Animated,
          title: "Starting index…",
          message: params.path ?? "All watch directories",
        });
        setView({ kind: "progress", startWith: params });
      }}
    />
  );
}

function FormView({
  isLoading,
  bootError,
  onStart,
}: {
  isLoading: boolean;
  bootError: string | null;
  onStart: (params: IndexParams) => void;
}) {
  // Belt-and-suspenders: prevent the action from firing twice in the same form
  // session if the user mashes the submit shortcut before the view swap renders.
  const fired = useRef(false);

  return (
    <Form
      isLoading={isLoading}
      navigationTitle="Index Files"
      actions={
        <ActionPanel>
          <Action.SubmitForm
            title="Start Indexing"
            icon={Icon.Play}
            onSubmit={(values: { paths: string[]; fileTypes: string[] }) => {
              if (fired.current) return;
              fired.current = true;
              onStart({
                path: values.paths?.[0],
                fileTypes: values.fileTypes ?? [],
              });
            }}
          />
        </ActionPanel>
      }
    >
      {bootError && (
        <Form.Description title="Server not reachable" text={bootError} />
      )}
      <Form.Description text="Pick a folder (or a single file) to index. Leave empty to index every configured watch directory. Indexing can take many minutes for large folders." />
      <Form.FilePicker
        id="paths"
        title="Path"
        allowMultipleSelection={false}
        canChooseDirectories
        canChooseFiles
      />
      <Form.TagPicker id="fileTypes" title="File Types" placeholder="All types">
        {FILE_TYPES.map((t) => (
          <Form.TagPicker.Item key={t} value={t} title={t} />
        ))}
      </Form.TagPicker>
    </Form>
  );
}

function ProgressView({
  startWith,
  onRunAnother,
}: {
  startWith?: IndexParams;
  onRunAnother: () => void;
}) {
  const [status, setStatus] = useState<IndexStatus | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [phase, setPhase] = useState<"starting" | "running" | "done" | "failed">(
    startWith ? "starting" : "running",
  );
  const [now, setNow] = useState(() => Date.now() / 1000);
  const triggered = useRef(false);

  useEffect(() => {
    let cancelled = false;
    let pollTimer: ReturnType<typeof setTimeout> | null = null;
    let terminal = false;

    const schedule = (delay: number) => {
      if (cancelled || terminal) return;
      pollTimer = setTimeout(poll, delay);
    };

    const poll = async () => {
      if (cancelled || terminal) return;
      try {
        const s = await getIndexStatus();
        if (cancelled) return;
        setStatus(s);
        setPollError(null);

        if (s.running) {
          setPhase("running");
          schedule(2000);
        } else if (s.error) {
          terminal = true;
          setPhase("failed");
        } else if (s.finished_at) {
          terminal = true;
          setPhase("done");
          showToast({
            style: Toast.Style.Success,
            title: "Indexing complete",
            message: s.stats
              ? `Indexed ${s.stats.indexed}, skipped ${s.stats.skipped}`
              : undefined,
          });
        } else {
          // No job state yet — server might not have picked up our POST.
          // Keep polling briefly.
          schedule(2000);
        }
      } catch (error) {
        if (cancelled) return;
        setPollError(error instanceof Error ? error.message : "Unknown error");
        schedule(5000);
      }
    };

    const start = async () => {
      if (!startWith || triggered.current) return;
      triggered.current = true;
      try {
        await triggerIndex(startWith);
      } catch (error) {
        const code = (error as Error & { code?: number }).code;
        if (code === 409) return; // joining an existing job is fine
        if (cancelled) return;
        terminal = true;
        setPhase("failed");
        setPollError(
          error instanceof Error ? error.message : "Failed to start indexing",
        );
        showToast({
          style: Toast.Style.Failure,
          title: "Failed to start indexing",
          message: error instanceof Error ? error.message : "Unknown error",
        });
      }
    };

    start().finally(() => {
      if (!cancelled && !terminal) poll();
    });

    const tick = setInterval(() => setNow(Date.now() / 1000), 1000);

    return () => {
      cancelled = true;
      if (pollTimer) clearTimeout(pollTimer);
      clearInterval(tick);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const markdown = renderProgress(status, phase, pollError, now, startWith);
  const isLoading = phase === "starting" || phase === "running";

  return (
    <Detail
      isLoading={isLoading}
      navigationTitle={
        phase === "running" || phase === "starting"
          ? "Indexing…"
          : phase === "failed"
            ? "Index Failed"
            : "Index Complete"
      }
      markdown={markdown}
      actions={
        <ActionPanel>
          {!isLoading && (
            <Action title="Run Another Index" icon={Icon.Repeat} onAction={onRunAnother} />
          )}
          <Action
            title="Open Index Status Page"
            icon={Icon.Gauge}
            onAction={async () => {
              try {
                await launchCommand({ name: "status", type: LaunchType.UserInitiated });
              } catch {
                // ignore — dev-mode quirk if the command isn't deployed yet
              }
            }}
          />
          {status?.path && (
            <Action
              title="Reveal Indexed Path in Finder"
              icon={Icon.Finder}
              onAction={() => {
                if (status.path) open(status.path);
              }}
            />
          )}
        </ActionPanel>
      }
    />
  );
}

function renderProgress(
  status: IndexStatus | null,
  phase: "starting" | "running" | "done" | "failed",
  pollError: string | null,
  now: number,
  startWith?: IndexParams,
): string {
  const lines: string[] = [];

  if (phase === "starting") {
    lines.push("# Starting index…", "");
    if (startWith?.path) lines.push(`**Path:** \`${startWith.path}\``);
    else lines.push("**Path:** _all watch directories_");
    if (startWith?.fileTypes?.length) {
      lines.push(`**File types:** ${startWith.fileTypes.join(", ")}`);
    }
    lines.push("", "_Contacting server…_");
    if (pollError) lines.push("", `_Error: ${pollError} (will retry)_`);
    return lines.join("\n");
  }

  if (!status) {
    return pollError
      ? `# Cannot reach the server\n\n\`\`\`\n${pollError}\n\`\`\`\n\nMake sure the SemFile server is running: \`semfile serve\``
      : "_Loading…_";
  }

  const heading =
    phase === "running"
      ? "# Indexing…"
      : phase === "failed"
        ? "# Index failed"
        : "# Index complete";
  lines.push(heading, "");

  if (status.path) {
    lines.push(`**Path:** \`${status.path}\``);
  } else {
    lines.push("**Path:** _all watch directories_");
  }
  if (status.file_types?.length) {
    lines.push(`**File types:** ${status.file_types.join(", ")}`);
  }

  if (status.started_at) {
    const end = status.finished_at ?? now;
    const elapsed = Math.max(0, end - status.started_at);
    lines.push(`**Elapsed:** ${formatDuration(elapsed)}`);
  }

  if (status.count_at_start !== null) {
    const delta = status.count - status.count_at_start;
    lines.push(`**Files in index:** ${status.count} (+${delta} since start)`);
  } else {
    lines.push(`**Files in index:** ${status.count}`);
  }

  if (status.stats) {
    lines.push("", "## Stats", "");
    lines.push(`- Scanned: ${status.stats.scanned}`);
    lines.push(`- Indexed: ${status.stats.indexed}`);
    lines.push(`- Skipped: ${status.stats.skipped}`);
    lines.push(`- Failed: ${status.stats.failed}`);
    lines.push(`- Removed: ${status.stats.removed}`);
  }

  if (status.error) {
    lines.push("", "## Error", "", "```", status.error, "```");
  }

  if (pollError) {
    lines.push("", `_Polling error: ${pollError} (will retry)_`);
  }

  return lines.join("\n");
}

function formatDuration(seconds: number): string {
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
