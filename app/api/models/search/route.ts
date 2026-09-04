const CATALOG_ORIGIN = 'https://voice-models.com';

function decodeHtml(value: string) {
  return value
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)))
    .replace(/&#x([0-9a-f]+);/gi, (_, code) => String.fromCharCode(Number.parseInt(code, 16)))
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#039;|&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&nbsp;/g, ' ');
}

function plainText(value: string) {
  return decodeHtml(value.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim());
}

function absoluteUrl(value: string) {
  try {
    const url = new URL(decodeHtml(value), CATALOG_ORIGIN);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : '';
  } catch {
    return '';
  }
}

function attribute(tag: string, name: string) {
  const match = tag.match(new RegExp(`${name}=["']([^"']*)["']`, 'i'));
  return match ? decodeHtml(match[1]) : '';
}

function parseResults(table: string) {
  const rows = table.match(/<tr\b[\s\S]*?<\/tr>/gi) ?? [];
  return rows.flatMap((row, index) => {
    const links = [...row.matchAll(/<a\b[^>]*href=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi)].map((match) => ({
      href: absoluteUrl(match[1]),
      label: plainText(match[2]),
      tag: match[0],
    }));
    const model = links.find((link) => /voice-models\.com\/model\//i.test(link.href));
    const download = links.find((link) =>
      !/voice-models\.com|easyaivoice\.com/i.test(link.href) &&
      /huggingface\.co|drive\.google\.com|mega\.nz|pixeldrain\.com|mediafire\.com|weights\.gg/i.test(link.href),
    );
    if (!model || !download) return [];

    const creatorLink = links.find((link) => /voice-models\.com\/creator\//i.test(link.href));
    const button = row.match(/<button\b[^>]*data-sample-name=["'][^"']+["'][^>]*>/i)?.[0] ?? '';
    const sampleName = attribute(button, 'data-sample-name');
    const sampleReady = attribute(button, 'data-sample-ready') === '1';
    const modelId = attribute(button, 'data-model-id') || model.href.split('/').pop() || String(index);
    const size = plainText(row).match(/\b\d+(?:\.\d+)?\s*(?:KB|MB|GB)\b/i)?.[0];

    return [{
      id: modelId,
      name: model.label || `Community voice ${index + 1}`,
      pageUrl: model.href,
      downloadUrl: download.href,
      creator: creatorLink?.label || undefined,
      size,
      engineReady: !/mega\.nz|mediafire\.com/i.test(download.href),
      sampleUrl: sampleReady && sampleName
        ? `https://s3.us-west-000.backblazeb2.com/bbvoe3/samples2/${encodeURIComponent(sampleName)}.mp3`
        : undefined,
    }];
  }).slice(0, 18);
}

export async function GET(request: Request) {
  const query = new URL(request.url).searchParams.get('q')?.trim() ?? '';
  if (query.length < 2 || query.length > 100) {
    return Response.json({ error: 'Enter between 2 and 100 characters.' }, { status: 400 });
  }

  try {
    const response = await fetch(`${CATALOG_ORIGIN}/fetch_data.php`, {
      method: 'POST',
      headers: {
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'x-requested-with': 'XMLHttpRequest',
        referer: `${CATALOG_ORIGIN}/`,
      },
      body: new URLSearchParams({ page: '1', search: query }).toString(),
      signal: AbortSignal.timeout(12000),
    });
    if (!response.ok) throw new Error(`Catalog returned ${response.status}`);
    const payload = await response.json() as { table?: string };
    const results = parseResults(payload.table ?? '');
    return Response.json({ query, results, source: CATALOG_ORIGIN }, {
      headers: { 'cache-control': 'public, max-age=120, s-maxage=300' },
    });
  } catch {
    return Response.json(
      { error: 'The community catalog is not responding. Try again in a moment or open it directly.' },
      { status: 502 },
    );
  }
}
