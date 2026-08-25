/**
 * Container-backed SandboxProvider.
 *
 * One container per sandbox, addressed by the absolute working directory inside
 * it. That choice matters for two reasons:
 *
 *  - The provider contract requires `pwd` to print the sandbox id, so the id has
 *    to *be* a path rather than a container name.
 *  - An absolute id activates the path-id branch of the shared contract suite,
 *    which asserts that one sandbox cannot reach a sibling by `../` or by
 *    absolute path. Separate containers satisfy that structurally: the sibling
 *    path does not exist in the other mount namespace.
 *
 * Unlike the bubblewrap-based local provider, a container can be given a GPU.
 * The NVIDIA driver refuses to initialise inside an unprivileged user namespace,
 * which is what bubblewrap creates, so bind-mounting `/dev/nvidia*` into a bwrap
 * jail yields `CUDA_ERROR_OPERATING_SYSTEM` no matter how the policy is widened.
 * The container toolkit exists to solve exactly that, so `deviceRequests` here
 * maps onto `--gpus`.
 *
 * File transfer goes through `docker exec` with a piped stdin/stdout rather than
 * encoding payloads into a command string. Encoding into argv caps out around
 * 96 KiB on `MAX_ARG_STRLEN` / `E2BIG` (see upstream issue #416), and a sandbox
 * that cannot accept a file larger than a small source file is not much use.
 */

import type {
  CodeModeTransport,
  ExecResult,
  SandboxBuild,
  SandboxExecParams,
  SandboxProvider,
} from '@truefoundry/trueforge-core/core';
import {
  absolutizeRelativeExecEnv,
  SandboxFileNotFoundError,
  SandboxFileTooLargeError,
  SandboxNotAvailableError,
  SandboxPathIsDirectoryError,
  shellEscape,
  validateNoPathTraversal,
} from '@truefoundry/trueforge-core/core';
import { spawn } from 'node:child_process';
import { posix, resolve, sep } from 'node:path';
import { ulid } from 'ulid';
import type { Logger } from 'winston';

/** Parent directory of every sandbox root inside the container. */
const SANDBOX_PARENT = '/sandbox';
const DEFAULT_EXEC_TIMEOUT_SECONDS = 60;
const DEFAULT_FILE_MAX_BYTES = 10 * 1024 * 1024;
/** Cap for `docker version` / `docker image inspect` probes, not general exec. */
const PROBE_TIMEOUT_MS = 10_000;
const CONTAINER_NAME_PREFIX = 'tfy-sbx-';

export interface DockerSandboxProviderOptions {
  /** Image the sandbox runs. Must provide a POSIX shell and `python3`. */
  image: string;
  logger: Logger;
  /** `docker` by default; set to `podman` or an absolute path to override. */
  dockerBinary?: string;
  /**
   * Value passed to `--gpus`, e.g. `all` or `device=0`. Omitted means no GPU,
   * which is the right default: attaching a GPU to a sandbox that does not need
   * one wastes a scarce device and slows container start.
   */
  gpus?: string;
  execTimeoutSeconds?: number;
  fileMaxBytesForDownload?: number;
  /**
   * Extra `docker run` arguments. Intended for read-only host mounts such as a
   * CUDA toolkit, so the image can stay small. Never interpolated through a
   * shell.
   */
  extraRunArgs?: readonly string[];
}

export type DockerSandboxSupportResult =
  | { supported: true; version: string }
  | { supported: false; reason: string };

interface RunResult {
  exitCode: number;
  stdout: Buffer;
  stderr: string;
  timedOut: boolean;
}

export class DockerSandboxProvider implements SandboxProvider {
  readonly type = 'docker';

  private readonly image: string;
  private readonly dockerBinary: string;
  private readonly gpus: string | undefined;
  private readonly execTimeoutSeconds: number;
  private readonly fileMaxBytesForDownload: number;
  private readonly extraRunArgs: readonly string[];
  private readonly logger: Logger;

  /** sandboxId (in-container path) -> container name. */
  private readonly containers = new Map<string, string>();
  /** Set while a background `docker pull` is running; cleared when it settles. */
  private pullInFlight: Promise<void> | undefined;
  /** Reason the last background pull failed, surfaced by getImageBuildStatus. */
  private pullFailure: string | undefined;

  private static readonly readyBuild: SandboxBuild = { status: 'ready', reason: null, metadata: null };

  constructor(options: DockerSandboxProviderOptions) {
    this.image = options.image;
    this.dockerBinary = options.dockerBinary ?? 'docker';
    this.gpus = options.gpus;
    this.execTimeoutSeconds = options.execTimeoutSeconds ?? DEFAULT_EXEC_TIMEOUT_SECONDS;
    this.fileMaxBytesForDownload = options.fileMaxBytesForDownload ?? DEFAULT_FILE_MAX_BYTES;
    this.extraRunArgs = options.extraRunArgs ?? [];
    this.logger = options.logger;
  }

  /** Probe whether a usable container runtime is present. */
  static async isSupported(dockerBinary = 'docker'): Promise<DockerSandboxSupportResult> {
    try {
      const result = await runProcess({
        file: dockerBinary,
        args: ['version', '--format', '{{.Server.Version}}'],
        timeoutMs: PROBE_TIMEOUT_MS,
      });
      if (result.exitCode !== 0) {
        return {
          supported: false,
          reason: `\`${dockerBinary} version\` exited ${String(result.exitCode)}: ${result.stderr.trim() || 'no stderr'}`,
        };
      }
      return { supported: true, version: result.stdout.toString('utf8').trim() };
    } catch (error) {
      return { supported: false, reason: `${dockerBinary} not usable: ${errorMessage(error)}` };
    }
  }

  /**
   * Ensures the image is present, pulling in the background if not.
   *
   * Must return promptly: callers wrap this in a short `withTimeout` (3s in the
   * settings route), and a cold CUDA image is several gigabytes. So the pull is
   * started detached and the call reports `pending`, which is exactly the
   * contract the interface documents.
   */
  async buildImage(): Promise<SandboxBuild> {
    if (await this.imagePresent()) {
      return DockerSandboxProvider.readyBuild;
    }
    if (this.pullInFlight === undefined) {
      this.pullFailure = undefined;
      this.pullInFlight = runProcess({
        file: this.dockerBinary,
        args: ['pull', this.image],
        timeoutMs: 60 * 60_000,
      })
        .then((pull) => {
          if (pull.exitCode !== 0) {
            this.pullFailure = pull.stderr.trim() || `exit ${String(pull.exitCode)}`;
          }
        })
        .catch((error: unknown) => {
          this.pullFailure = errorMessage(error);
        })
        .finally(() => {
          this.pullInFlight = undefined;
        });
      this.logger.info('DockerSandboxProvider started image pull', { image: this.image });
    }
    return { status: 'pending', reason: `pulling ${this.image}`, metadata: { image: this.image } };
  }

  async getImageBuildStatus(): Promise<SandboxBuild> {
    if (await this.imagePresent()) {
      return DockerSandboxProvider.readyBuild;
    }
    if (this.pullInFlight !== undefined) {
      return { status: 'pending', reason: `pulling ${this.image}`, metadata: { image: this.image } };
    }
    if (this.pullFailure !== undefined) {
      return {
        status: 'failed',
        reason: `failed to pull ${this.image}: ${this.pullFailure}`,
        metadata: { image: this.image },
      };
    }
    return { status: 'pending', reason: `image ${this.image} not present locally`, metadata: { image: this.image } };
  }

  private async imagePresent(): Promise<boolean> {
    const result = await runProcess({
      file: this.dockerBinary,
      args: ['image', 'inspect', this.image],
      timeoutMs: PROBE_TIMEOUT_MS,
    }).catch(() => undefined);
    return result?.exitCode === 0;
  }

  async createSandbox(): Promise<{ sandboxId: string }> {
    const id = ulid().toLowerCase();
    const containerName = `${CONTAINER_NAME_PREFIX}${id}`;
    const sandboxId = posix.join(SANDBOX_PARENT, id);

    const args = [
      'run',
      '--detach',
      '--name',
      containerName,
      // Keep the container alive without a workload; every command arrives via
      // `docker exec`. `sleep infinity` as PID 1 reaps nothing, so init is on.
      '--init',
      '--workdir',
      sandboxId,
      ...(this.gpus === undefined ? [] : ['--gpus', this.gpus]),
      ...this.extraRunArgs,
      this.image,
      'sleep',
      'infinity',
    ];

    const created = await runProcess({ file: this.dockerBinary, args, timeoutMs: 5 * 60_000 });
    if (created.exitCode !== 0) {
      throw new SandboxNotAvailableError(
        `failed to start sandbox container: ${created.stderr.trim() || 'no stderr'}`,
      );
    }

    this.containers.set(sandboxId, containerName);

    // `--workdir` creates the directory, but the layout subdirectories and the
    // venv do not exist yet. Failure here must not leak the container.
    try {
      await this.execInContainer({
        containerName,
        command: [
          `mkdir -p ${shellEscape(this.getToolResultDumpDir())}`,
          shellEscape(this.getFileUploadsDir()),
          shellEscape(this.getSkillsDir()),
        ].join(' '),
        cwd: sandboxId,
        timeoutSeconds: this.execTimeoutSeconds,
      });
    } catch (error) {
      await this.removeContainer(sandboxId).catch(() => undefined);
      throw error;
    }

    this.logger.info('DockerSandboxProvider created sandbox', {
      sandboxId,
      containerName,
      image: this.image,
      gpus: this.gpus ?? null,
    });
    return { sandboxId };
  }

  async exec(params: SandboxExecParams): Promise<ExecResult> {
    const containerName = this.requireContainer(params.sandboxId);
    try {
      const cwd =
        params.cwd === undefined || params.cwd === ''
          ? params.sandboxId
          : this.resolveInSandboxRoot(params.sandboxId, params.cwd);
      const env =
        params.env === undefined
          ? undefined
          : absolutizeRelativeExecEnv({ root: params.sandboxId, env: params.env });

      const result = await this.execInContainer({
        containerName,
        command: params.command,
        cwd,
        ...(env === undefined ? {} : { env }),
        timeoutSeconds: params.timeoutSeconds ?? this.execTimeoutSeconds,
      });

      return {
        success: true,
        response: {
          exitCode: result.exitCode,
          result: result.stdout.toString('utf8') + result.stderr,
        },
      };
    } catch (error) {
      if (error instanceof SandboxNotAvailableError) {
        throw error;
      }
      return { success: false, error: errorMessage(error) };
    }
  }

  getAdditionalInstructions(): string {
    return [
      'SANDBOX RULES:',
      `- Commands run inside a container from image ${this.image}.`,
      ...(this.gpus === undefined
        ? []
        : ['- An NVIDIA GPU is attached. `nvidia-smi` and CUDA are available.']),
      '- uploads, skills, and tool-results live in the sandbox working directory.',
      '- ALL file creation and writes MUST stay within the sandbox working directory.',
      '- The container is discarded when the sandbox ends; nothing outside the working directory persists.',
    ].join('\n');
  }

  // Cwd-relative, matching the local provider: exec cwd is the sandbox root, so
  // the layout paths stay free of absolute prefixes.
  getToolResultDumpDir(): string {
    return 'tool-results';
  }

  getGitCredentialsPath(): string {
    return '.git-credentials';
  }

  getFileUploadsDir(): string {
    return 'uploads';
  }

  getSkillsDir(): string {
    return 'skills';
  }

  getGitDownloaderPath(): string {
    return 'git_downloader.py';
  }

  async downloadFile(params: { sandboxId: string; path: string }): Promise<Buffer> {
    const containerName = this.requireContainer(params.sandboxId);
    const absolutePath = this.resolveInSandboxRoot(params.sandboxId, params.path);

    // Classify before transferring so the caller gets the specific error rather
    // than a shell diagnostic on stderr.
    const stat = await this.execInContainer({
      containerName,
      command: `if [ -d ${shellEscape(absolutePath)} ]; then echo DIR; elif [ -f ${shellEscape(absolutePath)} ]; then wc -c < ${shellEscape(absolutePath)}; else echo MISSING; fi`,
      cwd: params.sandboxId,
      timeoutSeconds: this.execTimeoutSeconds,
    });
    const verdict = stat.stdout.toString('utf8').trim();
    if (stat.exitCode !== 0 || verdict === 'MISSING') {
      throw new SandboxFileNotFoundError(params.path);
    }
    if (verdict === 'DIR') {
      throw new SandboxPathIsDirectoryError(params.path);
    }
    const size = Number.parseInt(verdict, 10);
    if (Number.isNaN(size)) {
      throw new SandboxFileNotFoundError(params.path);
    }
    if (size > this.fileMaxBytesForDownload) {
      throw new SandboxFileTooLargeError(params.path, size, this.fileMaxBytesForDownload);
    }

    const read = await this.execInContainer({
      containerName,
      command: `cat ${shellEscape(absolutePath)}`,
      cwd: params.sandboxId,
      timeoutSeconds: this.execTimeoutSeconds,
      maxStdoutBytes: this.fileMaxBytesForDownload,
    });
    if (read.exitCode !== 0) {
      throw new SandboxFileNotFoundError(params.path);
    }
    return read.stdout;
  }

  async uploadFile(params: { sandboxId: string; remotePath: string; content: Buffer }): Promise<void> {
    const containerName = this.requireContainer(params.sandboxId);
    const absolutePath = this.resolveInSandboxRoot(params.sandboxId, params.remotePath);
    const parent = posix.dirname(absolutePath);

    // Payload travels on stdin. Encoding it into the command string would cap
    // uploads at roughly 96 KiB (MAX_ARG_STRLEN); see upstream issue #416.
    const result = await this.execInContainer({
      containerName,
      command: `mkdir -p ${shellEscape(parent)} && cat > ${shellEscape(absolutePath)}`,
      cwd: params.sandboxId,
      timeoutSeconds: this.execTimeoutSeconds,
      stdin: params.content,
    });
    if (result.exitCode !== 0) {
      throw new Error(`upload to ${params.remotePath} failed: ${result.stderr.trim() || 'no stderr'}`);
    }
  }

  /**
   * Code Mode needs a bidirectional transport between the harness and the
   * sandbox. The local provider uses a unix socket on a shared filesystem, which
   * a container does not have by construction. Wiring this up needs a deliberate
   * transport choice, so it throws rather than half-working.
   */
  createCodeModeTransport(): CodeModeTransport {
    throw new Error('DockerSandboxProvider does not support Code Mode yet');
  }

  /** Removes every container this provider started. Safe to call twice. */
  async dispose(): Promise<void> {
    const ids = [...this.containers.keys()];
    await Promise.all(ids.map(async (id) => this.removeContainer(id).catch(() => undefined)));
  }

  private async removeContainer(sandboxId: string): Promise<void> {
    const containerName = this.containers.get(sandboxId);
    if (containerName === undefined) {
      return;
    }
    this.containers.delete(sandboxId);
    await runProcess({
      file: this.dockerBinary,
      args: ['rm', '--force', '--volumes', containerName],
      timeoutMs: 60_000,
    });
  }

  private requireContainer(sandboxId: string): string {
    const containerName = this.containers.get(sandboxId);
    if (containerName === undefined) {
      throw new SandboxNotAvailableError(`unknown sandbox ${sandboxId}`);
    }
    return containerName;
  }

  /**
   * Confine a caller-supplied path to the sandbox root. Uses the platform
   * resolver for `..` collapsing, then re-checks containment, because a path
   * that escapes must be reported as not-found rather than silently clamped.
   */
  private resolveInSandboxRoot(sandboxRootPath: string, userPath: string): string {
    validateNoPathTraversal(userPath);
    const resolved = userPath.startsWith('/')
      ? resolve(userPath)
      : resolve(sandboxRootPath, userPath);
    const root = resolve(sandboxRootPath);
    if (resolved !== root && !resolved.startsWith(root + sep)) {
      throw new SandboxFileNotFoundError(userPath);
    }
    return resolved;
  }

  private async execInContainer(params: {
    containerName: string;
    command: string;
    cwd: string;
    env?: Record<string, string>;
    timeoutSeconds: number;
    stdin?: Buffer;
    maxStdoutBytes?: number;
  }): Promise<RunResult> {
    const args = ['exec', '--workdir', params.cwd];
    if (params.stdin !== undefined) {
      args.push('--interactive');
    }
    for (const [key, value] of Object.entries(params.env ?? {})) {
      args.push('--env', `${key}=${value}`);
    }
    args.push(params.containerName, 'sh', '-c', params.command);

    const result = await runProcess({
      file: this.dockerBinary,
      args,
      timeoutMs: params.timeoutSeconds * 1000,
      ...(params.stdin === undefined ? {} : { stdin: params.stdin }),
      ...(params.maxStdoutBytes === undefined ? {} : { maxStdoutBytes: params.maxStdoutBytes }),
    });
    if (result.timedOut) {
      throw new Error(`command timed out after ${String(params.timeoutSeconds)}s`);
    }
    return result;
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Spawn a process, collecting stdout as bytes. stdout stays a Buffer because
 * downloads carry arbitrary binary content; stderr is decoded because it is only
 * ever shown to a human or a model.
 */
async function runProcess(params: {
  file: string;
  args: readonly string[];
  timeoutMs: number;
  stdin?: Buffer;
  maxStdoutBytes?: number;
}): Promise<RunResult> {
  return new Promise<RunResult>((resolvePromise, rejectPromise) => {
    const child = spawn(params.file, [...params.args], { stdio: ['pipe', 'pipe', 'pipe'] });

    const stdoutChunks: Buffer[] = [];
    let stdoutBytes = 0;
    let stderr = '';
    let timedOut = false;
    let settled = false;

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
    }, params.timeoutMs);

    child.stdout.on('data', (chunk: Buffer) => {
      stdoutBytes += chunk.length;
      if (params.maxStdoutBytes !== undefined && stdoutBytes > params.maxStdoutBytes) {
        child.kill('SIGKILL');
        return;
      }
      stdoutChunks.push(chunk);
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString('utf8');
    });

    const settle = (fn: () => void): void => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      fn();
    };

    child.on('error', (error) => {
      settle(() => rejectPromise(error));
    });

    child.on('close', (code) => {
      settle(() =>
        resolvePromise({
          exitCode: code ?? -1,
          stdout: Buffer.concat(stdoutChunks),
          stderr,
          timedOut,
        }),
      );
    });

    if (params.stdin !== undefined) {
      child.stdin.end(params.stdin);
    } else {
      child.stdin.end();
    }
  });
}
