'use client';

import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type SyntheticEvent,
} from 'react';
import {
  ArrowRight,
  AudioLines,
  Check,
  ChevronDown,
  Download,
  ExternalLink,
  FileAudio,
  LoaderCircle,
  MessageSquareText,
  Pause,
  Play,
  Search,
  ShieldCheck,
  Sparkles,
  Split,
  Upload,
  WandSparkles,
  X,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

type LocalAudio = { file: File; url: string };
type ModelResult = {
  id: string;
  name: string;
  pageUrl: string;
  downloadUrl: string;
  creator?: string;
  size?: string;
  sampleUrl?: string;
  engineReady?: boolean;
};
type Speaker = {
  id: string;
  label: string;
  color: string;
  model?: ModelResult;
  converted?: LocalAudio;
  gain: number;
};
type Segment = { id: string; start: number; end: number; speakerId: string; confidence: number };
type AnalysisResult = {
  jobId: string;
  buffer: AudioBuffer;
  duration: number;
  sampleRate: number;
  speechCoverage: number;
  peaks: number[];
  segments: Segment[];
};
type EngineHealth = {
  state: 'checking' | 'ready' | 'offline';
  backend?: string;
  device?: string;
  message?: string;
};
type EngineJob = {
  job_id: string;
  status: 'queued' | 'processing' | 'analyzed' | 'completed' | 'failed';
  progress: number;
  tracks: Record<string, string>;
  speaker_errors: Record<string, string>;
  error?: string;
  stage?: string;
  active_speaker?: string;
  downloaded_bytes?: number;
  total_bytes?: number;
};
type ToolRegistration = {
  name: string;
  title?: string;
  description: string;
  inputSchema: Record<string, unknown>;
  annotations?: Record<string, boolean>;
  execute: (input: unknown) => unknown;
};

declare global {
  interface Document {
    modelContext?: {
      registerTool: (
        tool: ToolRegistration,
        options?: { signal?: AbortSignal },
      ) => void | Promise<void>;
    };
  }
}

const COLORS = ['#d8ff55', '#b58cff', '#58d7ff', '#ff8cad', '#ffbd5c', '#77e6a5'];
const MODEL_CATALOG = 'https://voice-models.com/';
const MAX_SPEAKERS = 6;

function engineTrackUrl(path: string) {
  return `/api/engine/${path.replace(/^\/?api\/v1\//, '')}`;
}

function wait(milliseconds: number) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function readEngineHealth(): Promise<EngineHealth> {
  try {
    const response = await fetch('/api/engine/health', { cache: 'no-store' });
    const payload = await response.json() as { backend?: string; device?: string; error?: string; detail?: string };
    if (!response.ok) throw new Error(payload.error || payload.detail || 'The RVC engine is offline.');
    return { state: 'ready', backend: payload.backend, device: payload.device };
  } catch (error) {
    return { state: 'offline', message: error instanceof Error ? error.message : 'The RVC engine is offline.' };
  }
}

function makeSpeakers(count: number, current: Speaker[] = []) {
  return Array.from({ length: count }, (_, index) =>
    current[index] ?? {
      id: `speaker-${index + 1}`,
      label: `Voice ${String.fromCharCode(65 + index)}`,
      color: COLORS[index],
      gain: 1,
    },
  );
}

function formatTime(value: number) {
  const minutes = Math.floor(value / 60);
  const seconds = Math.max(0, value - minutes * 60);
  return `${minutes}:${seconds.toFixed(1).padStart(4, '0')}`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function jobProgressLabel(job: EngineJob, speakers: Speaker[]) {
  const speaker = speakers.find((candidate) => candidate.id === job.active_speaker);
  const voice = speaker?.label ?? 'the current voice';
  if (job.stage === 'downloading_model') {
    const received = job.downloaded_bytes === undefined ? '' : formatBytes(job.downloaded_bytes);
    const total = job.total_bytes ? ` / ${formatBytes(job.total_bytes)}` : '';
    return `Downloading ${voice} model${received ? ` · ${received}${total}` : ''}`;
  }
  if (job.stage === 'extracting_model') return `Unpacking ${voice} model`;
  if (job.stage === 'validating_model') return `Safety-checking ${voice} model`;
  if (job.stage === 'resolving_model') return `Connecting to ${voice} model`;
  if (job.stage === 'converting_speech') return `Converting ${voice} on the GPU`;
  if (job.stage === 'assembling_speaker_stem') return `Restoring ${voice} to the scene timeline`;
  if (job.stage === 'preparing_vocals') return 'Reading the isolated speech track';
  if (job.stage === 'mixing_master') return 'Combining voices with the stereo background';
  return `Converting the cast · ${Math.round(job.progress * 100)}%`;
}

function safeHttpUrl(value: string) {
  try {
    const url = new URL(value);
    return url.protocol === 'http:' || url.protocol === 'https:' ? url.href : undefined;
  } catch {
    return undefined;
  }
}

function replaceAudio(file: File, current?: LocalAudio) {
  if (current) URL.revokeObjectURL(current.url);
  return { file, url: URL.createObjectURL(file) };
}

function segmentFeatures(data: Float32Array, sampleRate: number, start: number, end: number) {
  const from = Math.max(0, Math.floor(start * sampleRate));
  const to = Math.min(data.length, Math.ceil(end * sampleRate));
  const step = Math.max(1, Math.floor((to - from) / 9000));
  let energy = 0;
  let crossings = 0;
  let difference = 0;
  let peak = 0;
  let previous = data[from] ?? 0;
  let samples = 0;
  for (let index = from; index < to; index += step) {
    const value = data[index];
    energy += value * value;
    difference += Math.abs(value - previous);
    if ((value >= 0) !== (previous >= 0)) crossings += 1;
    peak = Math.max(peak, Math.abs(value));
    previous = value;
    samples += 1;
  }
  return [
    Math.sqrt(energy / Math.max(1, samples)),
    crossings / Math.max(1, samples),
    difference / Math.max(1, samples),
    peak,
    Math.min(1, (end - start) / 8),
  ];
}

function clusterSegments(features: number[][], count: number) {
  if (!features.length) return [];
  const k = Math.max(1, Math.min(count, features.length));
  const dimensions = features[0].length;
  const mins = Array.from({ length: dimensions }, (_, d) => Math.min(...features.map((f) => f[d])));
  const maxs = Array.from({ length: dimensions }, (_, d) => Math.max(...features.map((f) => f[d])));
  const normalized = features.map((feature) =>
    feature.map((value, d) => (value - mins[d]) / Math.max(0.000001, maxs[d] - mins[d])),
  );
  let centers = Array.from({ length: k }, (_, index) => normalized[Math.floor((index * normalized.length) / k)]);
  let assignments = normalized.map((_, index) => index % k);
  for (let pass = 0; pass < 12; pass += 1) {
    assignments = normalized.map((feature) => {
      let best = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      centers.forEach((center, centerIndex) => {
        const distance = center.reduce((sum, value, d) => sum + (value - feature[d]) ** 2, 0);
        if (distance < bestDistance) {
          best = centerIndex;
          bestDistance = distance;
        }
      });
      return best;
    });
    centers = centers.map((center, centerIndex) => {
      const members = normalized.filter((_, index) => assignments[index] === centerIndex);
      if (!members.length) return center;
      return center.map((_, d) => members.reduce((sum, member) => sum + member[d], 0) / members.length);
    });
  }
  return assignments;
}

function monoSamples(buffer: AudioBuffer) {
  const mono = new Float32Array(buffer.length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let index = 0; index < data.length; index += 1) mono[index] += data[index] / buffer.numberOfChannels;
  }
  return mono;
}

function assignSegments(buffer: AudioBuffer, regions: Array<{ id: string; start: number; end: number }>, speakers: Speaker[]) {
  const mono = monoSamples(buffer);
  const features = regions.map((region) => segmentFeatures(mono, buffer.sampleRate, region.start, region.end));
  const assignments = clusterSegments(features, speakers.length);
  return regions.map((region, index) => ({
    ...region,
    speakerId: speakers[assignments[index] ?? 0].id,
    confidence: Math.max(0.46, Math.min(0.89, 0.62 + features[index][0] * 1.8)),
  }));
}

async function analyzeScene(file: File, speakers: Speaker[]): Promise<AnalysisResult> {
  const decoder = new AudioContext();
  try {
    const form = new FormData();
    form.append('audio_file', file, file.name);
    const [buffer, response] = await Promise.all([
      decoder.decodeAudioData(await file.arrayBuffer()),
      fetch('/api/engine/analyze', { method: 'POST', body: form }),
    ]);
    const payload = await response.json() as {
      job_id?: string;
      duration_ms?: number;
      sample_rate?: number;
      speech_coverage?: number;
      cues?: Array<{ cue_id: string; start_ms: number; end_ms: number }>;
      error?: string;
      detail?: string;
    };
    if (!response.ok || !payload.job_id || !payload.cues) {
      throw new Error(payload.error || payload.detail || 'The local engine could not analyze this scene.');
    }
    const regions = payload.cues.map((cue) => ({
      id: cue.cue_id,
      start: cue.start_ms / 1000,
      end: cue.end_ms / 1000,
    }));
    const segments = assignSegments(buffer, regions, speakers);
    const mono = monoSamples(buffer);

    const peakCount = 180;
    const peakSize = Math.max(1, Math.floor(mono.length / peakCount));
    const peaks = Array.from({ length: peakCount }, (_, peakIndex) => {
      let peak = 0;
      const from = peakIndex * peakSize;
      const to = Math.min(mono.length, from + peakSize);
      for (let index = from; index < to; index += Math.max(1, Math.floor(peakSize / 150))) {
        peak = Math.max(peak, Math.abs(mono[index]));
      }
      return peak;
    });

    return {
      jobId: payload.job_id,
      buffer,
      duration: (payload.duration_ms ?? Math.round(buffer.duration * 1000)) / 1000,
      sampleRate: payload.sample_rate ?? 48_000,
      speechCoverage: payload.speech_coverage ?? 0,
      peaks,
      segments,
    };
  } finally {
    await decoder.close();
  }
}

function UploadScene({ value, busy, onFile }: { value?: LocalAudio; busy: boolean; onFile: (file: File) => void }) {
  const inputId = useId();
  return (
    <label className={`scene-drop ${value ? 'has-file' : ''}`} htmlFor={inputId}>
      <input
        id={inputId}
        className="sr-only"
        type="file"
        accept="audio/mpeg,audio/wav,audio/x-wav,audio/mp4,audio/aac,.mp3,.wav,.m4a"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
        }}
      />
      <span className="drop-icon">{busy ? <LoaderCircle className="spin" /> : value ? <Check /> : <Upload />}</span>
      <span className="min-w-0 flex-1">
        <strong>{busy ? 'Separating vocals + finding speech…' : value ? value.file.name : 'Drop in the movie, scene, or song MP3'}</strong>
        <small>{value ? `${formatBytes(value.file.size)} · analyzed by your configured engine` : 'We isolate speech, then make a speaker draft for you to approve.'}</small>
      </span>
      <span className="drop-action">{value ? 'Replace' : 'Choose MP3'}</span>
    </label>
  );
}

export default function VoiceCastStudio() {
  const [speakers, setSpeakers] = useState<Speaker[]>(() => makeSpeakers(3));
  const [scene, setScene] = useState<LocalAudio>();
  const [analysis, setAnalysis] = useState<AnalysisResult>();
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState('');
  const [activeSpeakerId, setActiveSpeakerId] = useState('speaker-1');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<ModelResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [engineHealth, setEngineHealth] = useState<EngineHealth>({ state: 'checking' });
  const [engineJob, setEngineJob] = useState<EngineJob>();
  const [conversionState, setConversionState] = useState<'idle' | 'working' | 'done' | 'error'>('idle');
  const [conversionError, setConversionError] = useState('');
  const [mixState, setMixState] = useState<'idle' | 'working' | 'done' | 'error'>('idle');
  const [mixError, setMixError] = useState('');
  const [mix, setMix] = useState<{ url: string; duration: number }>();
  const [prompt, setPrompt] = useState('');
  const [promptReply, setPromptReply] = useState('');
  const [playingSegment, setPlayingSegment] = useState('');
  const [playhead, setPlayhead] = useState(0);
  const audioRef = useRef<HTMLAudioElement>(null);
  const stopTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const usedSpeakerIds = useMemo(
    () => new Set((analysis?.segments ?? []).map((segment) => segment.speakerId)),
    [analysis],
  );
  const usedSpeakers = speakers.filter((speaker) => usedSpeakerIds.has(speaker.id));
  const modelsReady = usedSpeakers.length > 0 && usedSpeakers.every((speaker) => speaker.model);

  useEffect(() => {
    void readEngineHealth().then(setEngineHealth);
  }, []);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    const registration = context.registerTool(
      {
        name: 'assign_character_models',
        title: 'Assign character voice models',
        description: 'Assign one to six named character models to the visible VoiceMerge speaker board.',
        inputSchema: {
          type: 'object',
          properties: {
            speakers: {
              type: 'array', minItems: 1, maxItems: MAX_SPEAKERS,
              items: {
                type: 'object',
                properties: {
                  label: { type: 'string' }, modelName: { type: 'string' },
                  modelPageUrl: { type: 'string', format: 'uri' },
                  downloadUrl: { type: 'string', format: 'uri' },
                },
                required: ['label', 'modelName', 'modelPageUrl', 'downloadUrl'], additionalProperties: false,
              },
            },
          },
          required: ['speakers'], additionalProperties: false,
        },
        annotations: { readOnlyHint: false, untrustedContentHint: false },
        execute(input) {
          const values = (input as { speakers?: unknown })?.speakers;
          if (!Array.isArray(values) || values.length < 1 || values.length > MAX_SPEAKERS) {
            throw new Error(`Provide between 1 and ${MAX_SPEAKERS} speakers.`);
          }
          const next = values.map((raw, index) => {
            const item = raw as Record<string, unknown>;
            if (![item.label, item.modelName, item.modelPageUrl, item.downloadUrl].every((value) => typeof value === 'string')) {
              throw new Error(`Speaker ${index + 1} is incomplete.`);
            }
            if (!safeHttpUrl(item.modelPageUrl as string) || !safeHttpUrl(item.downloadUrl as string)) {
              throw new Error(`Speaker ${index + 1} needs valid HTTP URLs.`);
            }
            return {
              id: `speaker-${index + 1}`, label: item.label as string, color: COLORS[index], gain: 1,
              model: {
                id: `tool-${index}`, name: item.modelName as string,
                pageUrl: item.modelPageUrl as string, downloadUrl: item.downloadUrl as string,
              },
            };
          });
          setSpeakers(next);
          setActiveSpeakerId(next[0].id);
          return { assigned: true, speakerCount: next.length };
        },
      },
      { signal: lifecycle.signal },
    );
    void Promise.resolve(registration).catch(() => undefined);
    return () => lifecycle.abort();
  }, []);

  async function runAnalysis(file: File, nextSpeakers = speakers) {
    setAnalyzing(true);
    setAnalysisError('');
    try {
      setAnalysis(await analyzeScene(file, nextSpeakers));
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : 'The local engine could not analyze that audio file.');
    } finally {
      setAnalyzing(false);
    }
  }

  function chooseScene(file: File) {
    const next = replaceAudio(file, scene);
    if (mix) URL.revokeObjectURL(mix.url);
    setScene(next);
    setAnalysis(undefined);
    setMix(undefined);
    setEngineJob(undefined);
    setSpeakers((current) => current.map((speaker) => ({ ...speaker, converted: undefined })));
    setConversionState('idle');
    setMixState('idle');
    void runAnalysis(file);
  }

  function changeSpeakerCount(count: number) {
    const next = makeSpeakers(count, speakers).map((speaker) => ({ ...speaker, converted: undefined }));
    setSpeakers(next);
    setConversionState('idle');
    setMixState('idle');
    if (!next.some((speaker) => speaker.id === activeSpeakerId)) setActiveSpeakerId(next[0].id);
    setAnalysis((current) => current && {
      ...current,
      segments: assignSegments(
        current.buffer,
        current.segments.map(({ id, start, end }) => ({ id, start, end })),
        next,
      ),
    });
  }

  async function searchModels(event?: SyntheticEvent<HTMLFormElement>) {
    event?.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setSearchError('');
    try {
      const response = await fetch(`/api/models/search?q=${encodeURIComponent(query.trim())}`);
      const payload = (await response.json()) as { results?: ModelResult[]; error?: string };
      if (!response.ok) throw new Error(payload.error || 'Model search failed.');
      setResults(payload.results ?? []);
    } catch (error) {
      setSearchError(error instanceof Error ? error.message : 'Model search failed.');
      setResults([]);
    } finally {
      setSearching(false);
    }
  }

  function assignModel(model: ModelResult) {
    setSpeakers((current) => current.map((speaker) =>
      speaker.id === activeSpeakerId ? { ...speaker, model, converted: undefined } : speaker,
    ));
    setConversionState('idle');
    setMixState('idle');
  }

  function playSegment(segment: Segment) {
    const player = audioRef.current;
    if (!player) return;
    if (stopTimer.current) clearTimeout(stopTimer.current);
    player.currentTime = segment.start;
    void player.play();
    setPlayingSegment(segment.id);
    stopTimer.current = setTimeout(() => {
      player.pause();
      setPlayingSegment('');
    }, Math.max(100, (segment.end - segment.start) * 1000));
  }

  function updateSegmentSpeaker(id: string, speakerId: string) {
    setAnalysis((current) => current && {
      ...current,
      segments: current.segments.map((segment) => segment.id === id ? { ...segment, speakerId, confidence: 1 } : segment),
    });
    setSpeakers((current) => current.map((speaker) => ({ ...speaker, converted: undefined })));
    setConversionState('idle');
    setMixState('idle');
  }

  function splitAtPlayhead() {
    if (!analysis || playhead <= 0 || playhead >= analysis.duration) return;
    const target = analysis.segments.find((segment) => playhead > segment.start + 0.08 && playhead < segment.end - 0.08);
    if (!target) {
      setPromptReply('Move the player into a colored section first, then split.');
      return;
    }
    setAnalysis({
      ...analysis,
      segments: analysis.segments.flatMap((segment) => segment.id === target.id ? [
        { ...segment, id: `${segment.id}-a`, end: playhead },
        { ...segment, id: `${segment.id}-b`, start: playhead },
      ] : segment),
    });
    setSpeakers((current) => current.map((speaker) => ({ ...speaker, converted: undefined })));
    setConversionState('idle');
    setMixState('idle');
    setPromptReply(`Split the section at ${formatTime(playhead)}.`);
  }

  async function convertAndMerge() {
    if (!analysis || !scene || !modelsReady || engineHealth.state !== 'ready') return;
    setConversionState('working');
    setConversionError('');
    setMixError('');
    setMixState('working');
    setEngineJob(undefined);
    try {
      const timeline = usedSpeakers.map((speaker) => ({
        speaker_id: speaker.id,
        model_id: speaker.model!.downloadUrl,
        gain: speaker.gain,
        cues: analysis.segments
          .filter((segment) => segment.speakerId === speaker.id)
          .sort((a, b) => a.start - b.start)
          .map((segment) => ({
            cue_id: segment.id.replace(/[^A-Za-z0-9._-]/g, '-').slice(0, 64),
            start_ms: Math.max(0, Math.round(segment.start * 1000)),
            end_ms: Math.min(Math.round(analysis.duration * 1000), Math.round(segment.end * 1000)),
          })),
      }));
      const response = await fetch('/api/engine/convert', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: analysis.jobId,
          timeline,
          params: { pitch: 0, f0_method: 'rmvpe', index_rate: 0.75 },
        }),
      });
      const created = await response.json() as { job_id?: string; error?: string; detail?: string };
      if (!response.ok || !created.job_id) throw new Error(created.error || created.detail || 'The conversion job could not start.');

      let job: EngineJob | undefined;
      for (let attempt = 0; attempt < 3600; attempt += 1) {
        const statusResponse = await fetch(`/api/engine/jobs/${created.job_id}`, { cache: 'no-store' });
        const payload = await statusResponse.json() as EngineJob & { detail?: string };
        if (!statusResponse.ok) throw new Error(payload.detail || 'The conversion job disappeared.');
        job = payload;
        setEngineJob(job);
        if (job.status === 'completed' || job.status === 'failed') break;
        await wait(1000);
      }
      if (!job || !['completed', 'failed'].includes(job.status)) throw new Error('The conversion job timed out.');

      const returned = await Promise.all(usedSpeakers.map(async (speaker) => {
        const track = job!.tracks[speaker.id];
        if (!track) return speaker;
        const trackResponse = await fetch(engineTrackUrl(track));
        if (!trackResponse.ok) throw new Error(`The ${speaker.label} track could not be downloaded.`);
        const blob = await trackResponse.blob();
        const file = new File([blob], `${speaker.id}.wav`, { type: 'audio/wav' });
        return { ...speaker, converted: replaceAudio(file, speaker.converted) };
      }));
      const returnedById = new Map(returned.map((speaker) => [speaker.id, speaker]));
      const nextSpeakers = speakers.map((speaker) => returnedById.get(speaker.id) ?? speaker);
      setSpeakers(nextSpeakers);
      const failures = Object.entries(job.speaker_errors ?? {});
      const masterTrack = job.tracks.master;
      if (!masterTrack) throw new Error(job.error || 'The engine did not produce a final master.');
      const masterResponse = await fetch(engineTrackUrl(masterTrack));
      if (!masterResponse.ok) throw new Error('The final master could not be downloaded.');
      const masterBlob = await masterResponse.blob();
      const nextMix = { url: URL.createObjectURL(masterBlob), duration: analysis.duration };
      if (mix) URL.revokeObjectURL(mix.url);
      setMix(nextMix);
      setMixState('done');
      setConversionState(failures.length ? 'error' : 'done');
      if (failures.length) {
        setConversionError(failures.map(([speakerId, message]) => `${speakerId}: ${message}`).join(' · '));
      }
    } catch (error) {
      setConversionError(error instanceof Error ? error.message : 'The integrated voice conversion failed.');
      setConversionState('error');
      setMixState('error');
    }
  }

  function applyPrompt(value = prompt) {
    const text = value.trim().toLowerCase();
    if (!text) return;
    const swap = text.match(/swap\s+(?:voice\s+)?([a-f])\s+(?:and|with)\s+(?:voice\s+)?([a-f])/);
    const level = text.match(/(?:make|turn)\s+(?:voice\s+)?([a-f])\s+(louder|quieter|softer)/);
    const count = text.match(/(?:use|set|make)\s+([1-6])\s+(?:voices|speakers|characters)/);
    if (swap) {
      const first = swap[1].charCodeAt(0) - 97;
      const second = swap[2].charCodeAt(0) - 97;
      if (speakers[first] && speakers[second]) {
        setSpeakers((current) => current.map((speaker, index) => index === first
          ? { ...speaker, model: current[second].model, converted: undefined }
          : index === second ? { ...speaker, model: current[first].model, converted: undefined } : speaker));
        setConversionState('idle');
        setMixState('idle');
        setPromptReply(`Swapped the character models on ${swap[1].toUpperCase()} and ${swap[2].toUpperCase()}.`);
      }
    } else if (level) {
      const index = level[1].charCodeAt(0) - 97;
      const delta = level[2] === 'louder' ? 0.12 : -0.12;
      setSpeakers((current) => current.map((speaker, speakerIndex) => speakerIndex === index
        ? { ...speaker, gain: Math.max(0.25, Math.min(1.75, speaker.gain + delta)), converted: undefined } : speaker));
      setConversionState('idle');
      setMixState('idle');
      setPromptReply(`Adjusted voice ${level[1].toUpperCase()} by ${delta > 0 ? '+' : ''}${Math.round(delta * 100)}%.`);
    } else if (count) {
      changeSpeakerCount(Number(count[1]));
      setPromptReply(`Rebuilding the draft for ${count[1]} speakers.`);
    } else if (/split|cut here|new section/.test(text)) {
      splitAtPlayhead();
    } else {
      setPromptReply('Try “swap A and B”, “make B louder”, “use 4 speakers”, or “split at playhead”.');
    }
    setPrompt('');
  }

  const activeSpeaker = speakers.find((speaker) => speaker.id === activeSpeakerId) ?? speakers[0];

  return (
    <main className="vm-page">
      <div className="studio-grid" aria-hidden="true" />
      <header className="vm-header">
        <div className="brand-lockup">
          <span className="brand-mark"><AudioLines /></span>
          <span>VOICEMERGE<span className="brand-9000">9000</span></span>
          <span className="brand-divider" />
          <span className="brand-subtitle">COMMUNITY CUT</span>
        </div>
        <div className="local-pill"><ShieldCheck /> One scene · human-reviewed timing</div>
      </header>

      <div className="vm-shell">
        <section className="hero">
          <div>
            <p className="eyebrow"><span className="live-dot" /> SCENE IN · CHARACTERS OUT</p>
            <h1>Cast any voice.<span>Keep the scene.</span></h1>
          </div>
          <p>Upload once. We draft who speaks where. You fix anything wrong, give each speaker a character, then convert and merge the whole cast in one click.</p>
        </section>

        <nav className="step-rail" aria-label="Workflow">
          {['Upload scene', 'Review speakers', 'Cast voices', 'Convert + merge'].map((label, index) => {
            const done = index === 0 ? !!scene : index === 1 ? !!analysis : index === 2 ? modelsReady : mixState === 'done';
            return <a key={label} href={`#step-${index + 1}`} className={done ? 'step-done' : ''}>
              <span>{done ? <Check /> : index + 1}</span><strong>{label}</strong>
            </a>;
          })}
        </nav>

        <section id="step-1" className="work-card upload-card">
          <div className="section-heading">
            <span className="section-number">01</span>
            <div><p className="eyebrow">ONE FILE</p><h2>Upload the original scene</h2></div>
          </div>
          <UploadScene value={scene} busy={analyzing} onFile={chooseScene} />
          {scene ? <audio ref={audioRef} className="scene-player" controls src={scene.url} onTimeUpdate={(event) => setPlayhead(event.currentTarget.currentTime)}><track kind="captions" /></audio> : null}
          {analysisError ? <p className="error-box">{analysisError}</p> : null}
        </section>

        <section id="step-2" className="work-card">
          <div className="section-heading between">
            <div className="flex items-center gap-4">
              <span className="section-number">02</span>
            <div><p className="eyebrow">ISOLATED SPEECH DRAFT</p><h2>Check who talks where</h2></div>
            </div>
            <label className="speaker-count">Expected voices
              <select value={speakers.length} onChange={(event) => changeSpeakerCount(Number(event.target.value))}>
                {[1, 2, 3, 4, 5, 6].map((value) => <option value={value} key={value}>{value}</option>)}
              </select><ChevronDown />
            </label>
          </div>

          {!analysis ? (
            <div className="empty-stage"><WandSparkles /><strong>Your speaker map appears here</strong><span>Upload an MP3 above to start.</span></div>
          ) : (
            <>
              <div className="review-note"><Sparkles /> Demucs removed the background and Silero found the speech. Speaker labels are still an acoustic guess—click any wrong one before converting.</div>
              <div className="timeline">
                <button className="timeline-seek" aria-label="Audio timeline. Click to move the playhead." onKeyDown={(event) => {
                  if (event.key === 'ArrowLeft' && audioRef.current) audioRef.current.currentTime = Math.max(0, audioRef.current.currentTime - 1);
                  if (event.key === 'ArrowRight' && audioRef.current && analysis) audioRef.current.currentTime = Math.min(analysis.duration, audioRef.current.currentTime + 1);
                }} onClick={(event) => {
                  if (!audioRef.current || !analysis) return;
                  const box = event.currentTarget.getBoundingClientRect();
                  audioRef.current.currentTime = ((event.clientX - box.left) / box.width) * analysis.duration;
                }} />
                <div className="waveform" aria-hidden="true">{analysis.peaks.map((peak, index) => <span key={index} style={{ height: `${Math.max(4, peak * 100)}%` }} />)}</div>
                {analysis.segments.map((segment) => {
                  const speaker = speakers.find((item) => item.id === segment.speakerId) ?? speakers[0];
                  return <button
                    key={segment.id}
                    className="timeline-segment"
                    style={{ left: `${(segment.start / analysis.duration) * 100}%`, width: `${Math.max(0.28, ((segment.end - segment.start) / analysis.duration) * 100)}%`, background: speaker.color }}
                    onClick={(event) => { event.stopPropagation(); playSegment(segment); }}
                    aria-label={`Play ${speaker.label} from ${formatTime(segment.start)} to ${formatTime(segment.end)}`}
                    title={`${speaker.label} · ${formatTime(segment.start)}–${formatTime(segment.end)}`}
                  />;
                })}
                <i className="playhead" style={{ left: `${(playhead / analysis.duration) * 100}%` }} />
              </div>
              <div className="speaker-legend">{speakers.map((speaker) => <span key={speaker.id}><i style={{ background: speaker.color }} />{speaker.label}</span>)}</div>
              <div className="segment-toolbar"><span>{analysis.segments.length} speech sections · {Math.round(analysis.speechCoverage * 100)}% dialogue coverage · {formatTime(analysis.duration)} total</span><Button variant="outline" size="sm" onClick={splitAtPlayhead}><Split />Split at playhead</Button></div>
              <div className="segment-list">
                {analysis.segments.map((segment, index) => {
                  const speaker = speakers.find((item) => item.id === segment.speakerId) ?? speakers[0];
                  return <div className="segment-row" key={segment.id}>
                    <button className="mini-play" onClick={() => playSegment(segment)}>{playingSegment === segment.id ? <Pause /> : <Play />}</button>
                    <span className="segment-index">{String(index + 1).padStart(2, '0')}</span>
                    <span className="segment-time">{formatTime(segment.start)} → {formatTime(segment.end)}</span>
                    <span className="confidence">{segment.confidence === 1 ? 'human checked' : `${Math.round(segment.confidence * 100)}% guess`}</span>
                    <select value={segment.speakerId} onChange={(event) => updateSegmentSpeaker(segment.id, event.target.value)} style={{ borderColor: speaker.color }}>
                      {speakers.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
                    </select>
                  </div>;
                })}
              </div>
            </>
          )}
        </section>

        <section id="step-3" className="work-card">
          <div className="section-heading">
            <span className="section-number">03</span>
            <div><p className="eyebrow">SEARCH · PREVIEW · ASSIGN</p><h2>Give each voice a character</h2></div>
          </div>
          <div className="cast-layout">
            <aside className="speaker-stack">
              {speakers.map((speaker, index) => <button
                key={speaker.id}
                className={`speaker-card ${activeSpeakerId === speaker.id ? 'speaker-active' : ''}`}
                onClick={() => setActiveSpeakerId(speaker.id)}
                style={{ '--speaker': speaker.color } as CSSProperties}
              >
                <span className="speaker-avatar">{String.fromCharCode(65 + index)}</span>
                <span className="min-w-0 flex-1"><strong>{speaker.label}</strong><small>{speaker.model?.name ?? 'No character picked yet'}</small></span>
                {speaker.model ? <Check /> : <ArrowRight />}
              </button>)}
            </aside>
            <div className="model-browser">
              <div className="model-browser-top">
                <div><p>Picking for</p><strong style={{ color: activeSpeaker.color }}>{activeSpeaker.label}</strong></div>
                {activeSpeaker.model ? <button className="clear-model" onClick={() => {
                  setSpeakers((current) => current.map((speaker) => speaker.id === activeSpeaker.id ? { ...speaker, model: undefined, converted: undefined } : speaker));
                  setConversionState('idle');
                  setMixState('idle');
                }}><X /> clear</button> : null}
              </div>
              <form className="model-search" onSubmit={searchModels}>
                <Search />
                <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search SpongeBob, Batman, singer, game character…" />
                <Button type="submit" disabled={searching || !query.trim()}>{searching ? <LoaderCircle className="spin" /> : 'Search'}</Button>
              </form>
              <div className="catalog-credit">Community results from <a href={MODEL_CATALOG} target="_blank" rel="noreferrer">voice-models.com <ExternalLink /></a>. Preview and respect each model’s credits.</div>
              {searchError ? <p className="error-box">{searchError}</p> : null}
              {!results.length ? <div className="model-empty"><Search /><span>Search the model library, then click <b>Use for {activeSpeaker.label}</b>.</span></div> : (
                <div className="model-results">
                  {results.map((model) => <article key={model.id} className={activeSpeaker.model?.id === model.id ? 'model-picked' : ''}>
                    <div className="model-copy"><strong>{model.name}</strong><small>{[model.creator, model.size].filter(Boolean).join(' · ') || 'Community model'}</small></div>
                    {model.sampleUrl ? <audio controls preload="none" src={model.sampleUrl}><track kind="captions" /></audio> : null}
                    <a href={model.pageUrl} target="_blank" rel="noreferrer" title="Open model page"><ExternalLink /></a>
                    <Button size="sm" disabled={model.engineReady === false} onClick={() => assignModel(model)}>{model.engineReady === false ? 'Source not automatic' : activeSpeaker.model?.id === model.id ? <><Check /> Assigned</> : `Use for ${activeSpeaker.label}`}</Button>
                  </article>)}
                </div>
              )}
            </div>
          </div>
        </section>

        <section id="step-4" className="work-card">
          <div className="section-heading">
            <span className="section-number">04</span>
            <div><p className="eyebrow">INTEGRATED RVC</p><h2>Convert and merge the cast</h2></div>
          </div>
          <div className={`bridge-note engine-${engineHealth.state}`}>
            {engineHealth.state === 'checking' ? <LoaderCircle className="spin" /> : engineHealth.state === 'ready' ? <Check /> : <ShieldCheck />}
            <div>
              <strong>{engineHealth.state === 'ready' ? 'RVC engine ready' : engineHealth.state === 'checking' ? 'Checking the RVC engine…' : 'RVC engine not connected'}</strong>
              <span>{engineHealth.state === 'ready'
                ? `${engineHealth.backend === 'copy' ? 'Development test engine' : 'Official RVC'} · ${engineHealth.device ?? 'automatic device'} · models download and cache behind the app`
                : engineHealth.state === 'offline'
                  ? engineHealth.message
                  : 'Testing the private inference connection.'}</span>
            </div>
            {engineHealth.state === 'offline' ? <Button variant="outline" size="sm" onClick={() => {
              setEngineHealth({ state: 'checking' });
              void readEngineHealth().then(setEngineHealth);
            }}>Retry</Button> : null}
          </div>
          <div className="job-list">
            {usedSpeakers.length ? usedSpeakers.map((speaker) => {
              const speakerError = engineJob?.speaker_errors?.[speaker.id];
              const speakerReady = Boolean(speaker.converted || engineJob?.tracks?.[speaker.id]);
              const speakerActive = engineJob?.active_speaker === speaker.id;
              return <article className="job-card" key={speaker.id} style={{ '--speaker': speaker.color } as CSSProperties}>
                <div className="job-number">{String.fromCharCode(65 + speakers.indexOf(speaker))}</div>
                <div className="job-title"><strong>{speaker.model?.name ?? `${speaker.label} needs a model`}</strong><span>{analysis?.segments.filter((segment) => segment.speakerId === speaker.id).length ?? 0} timed sections</span></div>
                <div className={`job-status ${speakerError ? 'job-failed' : speakerReady ? 'job-complete' : ''}`}>
                  {speakerError ? <X /> : speakerReady ? <Check /> : speakerActive ? <LoaderCircle className="spin" /> : <span />}
                  {speakerError ? 'Needs attention' : speakerReady ? 'Converted' : speakerActive ? 'Working' : speaker.model ? 'Ready' : 'Pick model'}
                </div>
                {speakerError ? <p className="speaker-error">{speakerError}</p> : null}
                {speaker.converted ? <audio controls src={speaker.converted.url}><track kind="captions" /></audio> : null}
              </article>;
            }) : <div className="empty-stage compact"><FileAudio /><strong>No dialogue jobs yet</strong><span>Upload and review a scene first.</span></div>}
          </div>
          {conversionState === 'working' ? <div className="conversion-progress" aria-live="polite">
            <div><span style={{ width: `${Math.max(2, (engineJob?.progress ?? 0.02) * 100)}%` }} /></div>
            <p>{engineJob?.status === 'processing' ? jobProgressLabel(engineJob, speakers) : 'Starting the reviewed conversion…'}</p>
          </div> : null}
          <div className="merge-strip">
            <div><p className="eyebrow">FINAL MASTER</p><strong>{!usedSpeakers.length
              ? 'Review the speaker map first.'
              : !modelsReady
                ? `Pick ${usedSpeakers.filter((speaker) => !speaker.model).length} more character model(s).`
                : engineHealth.state !== 'ready'
                  ? 'Connect the included RVC engine to enable one-click conversion.'
                  : 'One click converts every cue, restores the timing, and builds the master.'}</strong></div>
            <Button className="merge-button" disabled={!modelsReady || engineHealth.state !== 'ready' || conversionState === 'working' || mixState === 'working'} onClick={() => void convertAndMerge()}>{conversionState === 'working' || mixState === 'working' ? <LoaderCircle className="spin" /> : <WandSparkles />} {mix ? 'Rebuild voices + mix' : 'Convert all + merge'}</Button>
          </div>
          {conversionError ? <p className="error-box">{conversionError}</p> : null}
          {mixError ? <p className="error-box">{mixError}</p> : null}
          {mix ? <div className="mix-result"><span><Check /></span><div><strong>New cast is ready</strong><small>{formatTime(mix.duration)} · 48 kHz · 24-bit WAV</small></div><audio controls src={mix.url}><track kind="captions" /></audio><a href={mix.url} download="voicemerge9000-final.wav"><Download /> Download WAV</a></div> : null}
        </section>

        <section className="prompt-bar">
          <div className="prompt-icon"><MessageSquareText /></div>
          <div className="prompt-main">
            <p className="eyebrow">PROMPT THE EDIT</p>
            <form onSubmit={(event) => { event.preventDefault(); applyPrompt(); }}>
              <Input value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Try: swap A and B, make B louder, use 4 speakers…" />
              <Button type="submit"><Sparkles /> Apply</Button>
            </form>
            <div className="suggestion-chips">{['Swap A and B', 'Make B louder', 'Use 4 speakers', 'Split at playhead'].map((suggestion) => <button key={suggestion} onClick={() => applyPrompt(suggestion)}>{suggestion}</button>)}</div>
            {promptReply ? <p className="prompt-reply">{promptReply}</p> : null}
          </div>
        </section>

        <footer>Open source · Local Demucs + Silero analysis · Audio stays with your configured VoiceMerge engine · Use fictional, stylized, or authorized voices</footer>
      </div>
    </main>
  );
}
