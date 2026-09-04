'use client';

import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
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
type AnalysisResult = { buffer: AudioBuffer; duration: number; peaks: number[]; segments: Segment[] };
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
const CONVERTER = 'https://easyaivoice.com/run';
const MAX_SPEAKERS = 6;

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

function getPercentile(values: number[], fraction: number) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];
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

async function analyzeScene(file: File, speakers: Speaker[], sensitivity = 1): Promise<AnalysisResult> {
  const decoder = new AudioContext();
  try {
    const buffer = await decoder.decodeAudioData(await file.arrayBuffer());
    const sampleRate = buffer.sampleRate;
    const mono = new Float32Array(buffer.length);
    for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
      const data = buffer.getChannelData(channel);
      for (let index = 0; index < data.length; index += 1) mono[index] += data[index] / buffer.numberOfChannels;
    }

    const frameSeconds = 0.08;
    const frameSize = Math.max(256, Math.floor(sampleRate * frameSeconds));
    const energy: number[] = [];
    for (let from = 0; from < mono.length; from += frameSize) {
      let sum = 0;
      const to = Math.min(mono.length, from + frameSize);
      for (let index = from; index < to; index += 4) sum += mono[index] * mono[index];
      energy.push(Math.sqrt(sum / Math.max(1, Math.ceil((to - from) / 4))));
    }

    const noise = getPercentile(energy, 0.2);
    const voice = getPercentile(energy, 0.72);
    const threshold = Math.max(0.006, noise + (voice - noise) * (0.3 / sensitivity));
    const raw: Array<{ start: number; end: number }> = [];
    let activeStart = -1;
    energy.forEach((value, index) => {
      const active = value >= threshold;
      if (active && activeStart < 0) activeStart = index;
      if ((!active || index === energy.length - 1) && activeStart >= 0) {
        raw.push({ start: activeStart * frameSeconds, end: Math.min(buffer.duration, (index + 1) * frameSeconds) });
        activeStart = -1;
      }
    });

    const merged: Array<{ start: number; end: number }> = [];
    raw.forEach((region) => {
      const previous = merged.at(-1);
      if (previous && region.start - previous.end < 0.34) previous.end = region.end;
      else merged.push({ ...region });
    });
    let regions = merged.filter((region) => region.end - region.start >= 0.24);
    if (!regions.length) regions = [{ start: 0, end: buffer.duration }];

    const features = regions.map((region) => segmentFeatures(mono, sampleRate, region.start, region.end));
    const assignments = clusterSegments(features, speakers.length);
    const segments = regions.map((region, index) => ({
      id: `segment-${Date.now()}-${index}`,
      ...region,
      speakerId: speakers[assignments[index] ?? 0].id,
      confidence: Math.max(0.46, Math.min(0.89, 0.62 + features[index][0] * 1.8)),
    }));

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

    return { buffer, duration: buffer.duration, peaks, segments };
  } finally {
    await decoder.close();
  }
}

function audioBufferTo24BitWav(buffer: AudioBuffer) {
  const channels = Math.min(2, buffer.numberOfChannels);
  const blockAlign = channels * 3;
  const output = new ArrayBuffer(44 + buffer.length * blockAlign);
  const view = new DataView(output);
  const write = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
  };
  write(0, 'RIFF');
  view.setUint32(4, output.byteLength - 8, true);
  write(8, 'WAVE');
  write(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 24, true);
  write(36, 'data');
  view.setUint32(40, output.byteLength - 44, true);
  const data = Array.from({ length: channels }, (_, channel) => buffer.getChannelData(channel));
  let offset = 44;
  for (let sample = 0; sample < buffer.length; sample += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const input = Math.max(-1, Math.min(1, data[channel][sample]));
      let value = input < 0 ? Math.round(input * 8388608) : Math.round(input * 8388607);
      if (value < 0) value += 16777216;
      view.setUint8(offset, value & 255);
      view.setUint8(offset + 1, (value >> 8) & 255);
      view.setUint8(offset + 2, (value >> 16) & 255);
      offset += 3;
    }
  }
  return new Blob([output], { type: 'audio/wav' });
}

function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
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
        <strong>{busy ? 'Finding the dialogue…' : value ? value.file.name : 'Drop in the movie, scene, or song MP3'}</strong>
        <small>{value ? `${formatBytes(value.file.size)} · never uploaded for analysis` : 'We make a local first-pass speaker map. You approve it.'}</small>
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
  const [renderingSpeaker, setRenderingSpeaker] = useState('');
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
  const returnsReady = usedSpeakers.length > 0 && usedSpeakers.every((speaker) => speaker.converted);

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

  async function runAnalysis(file: File, nextSpeakers = speakers, sensitivity = 1) {
    setAnalyzing(true);
    setAnalysisError('');
    try {
      setAnalysis(await analyzeScene(file, nextSpeakers, sensitivity));
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : 'This browser could not read that audio file.');
    } finally {
      setAnalyzing(false);
    }
  }

  function chooseScene(file: File) {
    const next = replaceAudio(file, scene);
    setScene(next);
    setMixState('idle');
    void runAnalysis(file);
  }

  function changeSpeakerCount(count: number) {
    const next = makeSpeakers(count, speakers);
    setSpeakers(next);
    if (!next.some((speaker) => speaker.id === activeSpeakerId)) setActiveSpeakerId(next[0].id);
    if (scene) void runAnalysis(scene.file, next);
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
      speaker.id === activeSpeakerId ? { ...speaker, model } : speaker,
    ));
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
    setPromptReply(`Split the section at ${formatTime(playhead)}.`);
  }

  async function exportSpeakerJob(speaker: Speaker) {
    if (!analysis || !scene) return;
    setRenderingSpeaker(speaker.id);
    try {
      const offline = new OfflineAudioContext(
        Math.min(2, analysis.buffer.numberOfChannels),
        Math.ceil(analysis.duration * 48000),
        48000,
      );
      const source = offline.createBufferSource();
      const gate = offline.createGain();
      source.buffer = analysis.buffer;
      gate.gain.setValueAtTime(0, 0);
      analysis.segments.filter((segment) => segment.speakerId === speaker.id).forEach((segment) => {
        const fade = Math.min(0.025, (segment.end - segment.start) / 4);
        gate.gain.setValueAtTime(0, Math.max(0, segment.start - fade));
        gate.gain.linearRampToValueAtTime(1, segment.start);
        gate.gain.setValueAtTime(1, Math.max(segment.start, segment.end - fade));
        gate.gain.linearRampToValueAtTime(0, segment.end);
      });
      source.connect(gate).connect(offline.destination);
      source.start(0);
      const stem = await offline.startRendering();
      downloadBlob(audioBufferTo24BitWav(stem), `voicemerge-${speaker.label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-input.wav`);
    } finally {
      setRenderingSpeaker('');
    }
  }

  function converterUrl(speaker: Speaker) {
    if (!speaker.model) return CONVERTER;
    const url = new URL(CONVERTER);
    url.searchParams.append('url[]', speaker.model.downloadUrl);
    url.searchParams.append('pitch[]', '0');
    url.searchParams.set('source', 'voicemerge9000');
    return url.href;
  }

  function importConverted(speakerId: string, event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setSpeakers((current) => current.map((speaker) =>
      speaker.id === speakerId ? { ...speaker, converted: replaceAudio(file, speaker.converted) } : speaker,
    ));
    setMixState('idle');
  }

  async function renderMix() {
    if (!analysis || !scene || !returnsReady) return;
    setMixState('working');
    setMixError('');
    const decoder = new AudioContext();
    try {
      const returns = await Promise.all(usedSpeakers.map(async (speaker) => ({
        speaker,
        buffer: await decoder.decodeAudioData(await speaker.converted!.file.arrayBuffer()),
      })));
      const sampleRate = 48000;
      const offline = new OfflineAudioContext(2, Math.ceil(analysis.duration * sampleRate), sampleRate);
      const master = offline.createGain();
      const limiter = offline.createDynamicsCompressor();
      limiter.threshold.value = -3;
      limiter.knee.value = 5;
      limiter.ratio.value = 16;
      limiter.attack.value = 0.003;
      limiter.release.value = 0.13;
      master.connect(limiter).connect(offline.destination);

      const bed = offline.createBufferSource();
      const bedGain = offline.createGain();
      bed.buffer = analysis.buffer;
      bedGain.gain.setValueAtTime(1, 0);
      [...analysis.segments].sort((a, b) => a.start - b.start).forEach((segment) => {
        bedGain.gain.setValueAtTime(1, Math.max(0, segment.start - 0.06));
        bedGain.gain.linearRampToValueAtTime(0.16, segment.start + 0.02);
        bedGain.gain.setValueAtTime(0.16, Math.max(segment.start + 0.02, segment.end - 0.03));
        bedGain.gain.linearRampToValueAtTime(1, Math.min(analysis.duration, segment.end + 0.08));
      });
      bed.connect(bedGain).connect(master);
      bed.start(0);

      returns.forEach(({ speaker, buffer }) => {
        const source = offline.createBufferSource();
        const gain = offline.createGain();
        source.buffer = buffer;
        gain.gain.value = speaker.gain;
        source.connect(gain).connect(master);
        source.start(0);
      });

      const rendered = await offline.startRendering();
      const next = { url: URL.createObjectURL(audioBufferTo24BitWav(rendered)), duration: analysis.duration };
      if (mix) URL.revokeObjectURL(mix.url);
      setMix(next);
      setMixState('done');
    } catch (error) {
      setMixError(error instanceof Error ? error.message : 'The browser could not merge those files.');
      setMixState('error');
    } finally {
      await decoder.close();
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
          ? { ...speaker, model: current[second].model }
          : index === second ? { ...speaker, model: current[first].model } : speaker));
        setPromptReply(`Swapped the character models on ${swap[1].toUpperCase()} and ${swap[2].toUpperCase()}.`);
      }
    } else if (level) {
      const index = level[1].charCodeAt(0) - 97;
      const delta = level[2] === 'louder' ? 0.12 : -0.12;
      setSpeakers((current) => current.map((speaker, speakerIndex) => speakerIndex === index
        ? { ...speaker, gain: Math.max(0.25, Math.min(1.75, speaker.gain + delta)) } : speaker));
      setPromptReply(`Adjusted voice ${level[1].toUpperCase()} by ${delta > 0 ? '+' : ''}${Math.round(delta * 100)}%.`);
    } else if (count) {
      changeSpeakerCount(Number(count[1]));
      setPromptReply(`Rebuilding the draft for ${count[1]} speakers.`);
    } else if (/more sensitive|find more|missed dialogue/.test(text) && scene) {
      void runAnalysis(scene.file, speakers, 1.28);
      setPromptReply('Running a more sensitive dialogue pass. Review the extra sections.');
    } else if (/conservative|less sensitive|fewer sections/.test(text) && scene) {
      void runAnalysis(scene.file, speakers, 0.78);
      setPromptReply('Running a more conservative dialogue pass.');
    } else {
      setPromptReply('Try “swap A and B”, “make B louder”, “use 4 speakers”, or “find more dialogue”.');
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
        <div className="local-pill"><ShieldCheck /> Audio analysis stays on this device</div>
      </header>

      <div className="vm-shell">
        <section className="hero">
          <div>
            <p className="eyebrow"><span className="live-dot" /> SCENE IN · CHARACTERS OUT</p>
            <h1>Cast any voice.<span>Keep the scene.</span></h1>
          </div>
          <p>Upload once. We draft who speaks where. You fix anything wrong, give each speaker a character, and merge the returned voices back on the same timeline.</p>
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
              <div><p className="eyebrow">LOCAL AUTO-DRAFT</p><h2>Check who talks where</h2></div>
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
              <div className="review-note"><Sparkles /> This is a fast acoustic guess, not identity recognition. Click any wrong label and fix it before converting.</div>
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
              <div className="segment-toolbar"><span>{analysis.segments.length} dialogue sections · {formatTime(analysis.duration)} total</span><Button variant="outline" size="sm" onClick={splitAtPlayhead}><Split />Split at playhead</Button></div>
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
                {activeSpeaker.model ? <button className="clear-model" onClick={() => setSpeakers((current) => current.map((speaker) => speaker.id === activeSpeaker.id ? { ...speaker, model: undefined } : speaker))}><X /> clear</button> : null}
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
                    <Button size="sm" onClick={() => assignModel(model)}>{activeSpeaker.model?.id === model.id ? <><Check /> Assigned</> : `Use for ${activeSpeaker.label}`}</Button>
                  </article>)}
                </div>
              )}
            </div>
          </div>
        </section>

        <section id="step-4" className="work-card">
          <div className="section-heading">
            <span className="section-number">04</span>
            <div><p className="eyebrow">TIMED VOICE JOBS</p><h2>Convert, return, and merge</h2></div>
          </div>
          <div className="bridge-note"><ShieldCheck /><div><strong>The honest current bridge</strong><span>VoiceMerge prepares perfectly timed WAV jobs. The online converter runs the RVC model; bring each result back here. A local one-click engine is the next open-source milestone.</span></div></div>
          <div className="job-list">
            {usedSpeakers.length ? usedSpeakers.map((speaker) => {
              const inputId = `return-${speaker.id}`;
              return <article className="job-card" key={speaker.id} style={{ '--speaker': speaker.color } as CSSProperties}>
                <div className="job-number">{String.fromCharCode(65 + speakers.indexOf(speaker))}</div>
                <div className="job-title"><strong>{speaker.model?.name ?? `${speaker.label} needs a model`}</strong><span>{analysis?.segments.filter((segment) => segment.speakerId === speaker.id).length ?? 0} timed sections</span></div>
                <div className="job-actions">
                  <Button variant="outline" size="sm" disabled={!speaker.model || renderingSpeaker === speaker.id} onClick={() => exportSpeakerJob(speaker)}>{renderingSpeaker === speaker.id ? <LoaderCircle className="spin" /> : <Download />} 1. Voice input</Button>
                  <a className={`converter-link ${!speaker.model ? 'disabled-link' : ''}`} href={speaker.model ? converterUrl(speaker) : undefined} target="_blank" rel="noreferrer"><ExternalLink /> 2. Open converter</a>
                  <input id={inputId} className="sr-only" type="file" accept="audio/*,.wav,.mp3,.m4a" onChange={(event) => importConverted(speaker.id, event)} />
                  <label className={speaker.converted ? 'return-ready' : ''} htmlFor={inputId}><Upload /> {speaker.converted ? 'Returned' : '3. Return voice'}</label>
                </div>
                {speaker.converted ? <audio controls src={speaker.converted.url}><track kind="captions" /></audio> : null}
              </article>;
            }) : <div className="empty-stage compact"><FileAudio /><strong>No dialogue jobs yet</strong><span>Upload and review a scene first.</span></div>}
          </div>
          <div className="merge-strip">
            <div><p className="eyebrow">FINAL MASTER</p><strong>{returnsReady ? 'All returned voices are lined up.' : `Return ${Math.max(0, usedSpeakers.filter((speaker) => !speaker.converted).length)} more voice file(s).`}</strong></div>
            <Button className="merge-button" disabled={!returnsReady || mixState === 'working'} onClick={renderMix}>{mixState === 'working' ? <LoaderCircle className="spin" /> : <AudioLines />} Merge scene</Button>
          </div>
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
            <div className="suggestion-chips">{['Swap A and B', 'Make B louder', 'Use 4 speakers', 'Find more dialogue'].map((suggestion) => <button key={suggestion} onClick={() => applyPrompt(suggestion)}>{suggestion}</button>)}</div>
            {promptReply ? <p className="prompt-reply">{promptReply}</p> : null}
          </div>
        </section>

        <footer>Open-source friendly · Browser-local audio analysis · Use fictional, stylized, or authorized voices only · Respect model credits and takedowns</footer>
      </div>
    </main>
  );
}
