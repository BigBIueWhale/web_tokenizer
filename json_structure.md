Here’s a “decoder ring” for `o200k_base.json`—what’s in it, how it’s structured, and how each part should be used to (re)implement `tiktoken.html` correctly.

Everything below is based on actually parsing the JSON you provided and computing statistics over it in Python.

---

## 0. High-level picture

The file `o200k_base.json` is a **single encoding definition** for tiktoken. At top level it has exactly three keys:

```json
{
  "pat_str": string,
  "special_tokens": { string: int, ... },
  "bpe_ranks": string
}
```

Conceptually:

* **`pat_str`** – A big regex string that defines how to split text into “pieces” before BPE merging.
* **`special_tokens`** – Map from special token *text* (e.g. `"<|endoftext|>"`) to integer token IDs.
* **`bpe_ranks`** – A compact, space-separated, base64-encoded description of the main BPE vocabulary, which you must decompress into a map of **bytes sequence → rank**.

In a correct implementation, **all tokenization logic in your HTML/worker should be derived from these three things and nothing else**.

---

## 1. `pat_str`: how text is initially segmented

### 1.1 Type and size

* Type: **string**
* Observed length: **364 characters**

Example (shortened; it’s much longer in reality):

```text
"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lo}\\p{M}]..."
```

It’s a **Unicode-aware regular expression** that uses `\p{L}`, `\p{N}`, etc. This is the same pattern used by official tiktoken implementations for GPT-4/GPT-4o style tokenizers.

### 1.2 How you should use it

In your worker:

* **JS type**: `string`
* Convert to `RegExp` exactly like js-tiktoken:

```js
const regex = new RegExp(patStrFromJson, "ug");
```

> Flags:
>
> * `u` – Unicode
> * `g` – Global (allows multiple `.exec` iterations)

### 1.3 Role in the algorithm

When you encode a string:

1. You run this regex over the **plain text** (excluding special tokens) to obtain **“pieces”**. Each `match[0]` becomes a candidate chunk for BPE.
2. For each piece, you:

   * Encode it to bytes via `TextEncoder` (UTF-8).
   * Then either directly map it via the BPE ranks or run the BPE merge algorithm.

**Why you need `pat_str`:**

* Without it, you’d be splitting text arbitrarily (e.g. per character), which would not match OpenAI’s official tokenization.
* `pat_str` is what makes punctuation, words, emojis, whitespace, etc. break into the same pieces the reference tokenizer uses before BPE merging.

You do **not** need to understand every sub-pattern in `pat_str` to use it correctly; you just need to compile it and feed it into the encoding loop exactly as-is.

---

## 2. `special_tokens`: special text → token IDs

### 2.1 Structure

`special_tokens` is a small dictionary:

```json
"special_tokens": {
  "<|endoftext|>": 199999,
  "<|endofprompt|>": 200018
}
```

* Type: `{ [string]: number }`
* Count: **2 special tokens**
* Keys (special token strings) and lengths:

  * `"<|endoftext|>"` – length **13**, ID **199999**
  * `"<|endofprompt|>"` – length **15**, ID **200018**

### 2.2 Numeric ranges & interaction with main vocab

From analysis of the BPE ranks (next section):

* BPE tokens cover ranks **0 .. 199,997** (inclusive).
* Special tokens have IDs:

  * `199999` and `200018`
* These IDs **do not overlap** with any BPE token ID.
* Overall, across BPE + specials:

  * Unique token IDs: **200,000**
  * Numeric min/max: **0** to **200,018**
  * There are **19 numeric gaps** (e.g. 199,998, 200,000–200,017) that are unused by this encoding.

So your decode / visualize logic must **not assume IDs are a contiguous 0..N range**. You always consult maps.

### 2.3 How to use `special_tokens` in your implementation

You should build:

#### 2.3.1 A forward map (already given)

```ts
const specialTokens: Record<string, number> = data.special_tokens;
```

Usage:

* To assign a token ID when your encoder sees a special token string in the text (in allowed contexts).

#### 2.3.2 An inverse map for decoding

You’ll want:

```ts
const inverseSpecialTokens: Record<number, Uint8Array> = {};
for (const [text, rank] of Object.entries(specialTokens)) {
  inverseSpecialTokens[rank] = textEncoder.encode(text); // UTF-8 bytes
}
```

Usage:

* During `decode(tokens)`, if a token ID is in `inverseSpecialTokens`, you decode it by returning these bytes (then later run `TextDecoder` on the merged byte array).

#### 2.3.3 A special-token regex

For encoding, you need to detect occurrences of special tokens in text to:

* either treat them as **atomic units** (when allowed), or
* reject/throw when they appear but are **disallowed**.

You build it by **escaping the token strings themselves**:

```ts
function escapeRegex(str: string) {
  return str.replace(/[\\^$*+?.()|[\]{}]/g, "\\$&");
}

const specialRegex = new RegExp(
  Object.keys(specialTokens).map(escapeRegex).join("|"),
  "g"
);
```

You then use this regex to:

* scan the text for special tokens,
* slice the text into: regular segments + special segments,
* and encode each part accordingly.

#### 2.3.4 Allowed / disallowed logic

The professional implementation takes:

```ts
encode(
  text: string,
  allowedSpecial: string[] | "all" = [],
  disallowedSpecial: string[] | "all" = "all",
)
```

You should:

* Use `special_tokens` keys to build **allowed** and **disallowed** sets.
* When `disallowedSpecial` is not empty:

  * Build a regex from the **disallowed** subset.
  * Scan the entire `text` and **throw** if any disallowed special appears.

**Why you need `special_tokens`:**

* To know which strings to treat as atomic control tokens.
* To map those strings to integers.
* To enforce safety / correctness constraints (e.g., erroring on disallowed specials).

---

## 3. `bpe_ranks`: compressed BPE vocabulary

This is the big one.

### 3.1 Raw structure

* Type: **string**
* Observed length: **2,325,049 characters**
* The string contains:

  * No `"\n"` (newline) characters.
  * Lots of `' '` (space) separators.

If we split on spaces:

```py
parts = bpe_ranks.split(" ")
len(parts) == 200_000
```

The beginning looks like:

```text
! 0 IQ== Ig== Iw== JA== JQ== Jg== Jw== KA== KQ== Kg== ...
```

So:

* `parts[0]` = `"!"` → a sentinel
* `parts[1]` = `"0"` → starting offset
* `parts[2:]` = `"IQ=="`, `"Ig=="`, `"Iw=="`, … → base64-encoded tokens

There is **exactly one `"!"` token** in the entire string, and only **one group**:

* No multiple segments like `"! 512 ..."`.
* It’s one sentinel, one offset, and then **199,998** Base64 tokens.

So we have:

```py
offset = int(parts[1])           # => 0
tokens = parts[2:]               # len(tokens) == 199_998
```

### 3.2 Token IDs (ranks) implied by `bpe_ranks`

For each base64 token `tokens[i]`:

* Token ID (rank) = `offset + i`

Given:

* `offset` = 0
* `len(tokens)` = 199,998

We get:

* BPE token IDs: **0 .. 199,997** (inclusive, contiguous).
* Number of BPE tokens: **199,998**.

Combined with `special_tokens`, there are **200,000 distinct token IDs**, but with numeric gaps as noted earlier.

### 3.3 Decoding Base64 tokens to bytes

Each base64 string represents a **raw bytes sequence** (not necessarily valid UTF-8).

Python decoding reveals:

* All tokens are valid base64; no invalid tokens.
* Number of unique base64 strings: **199,998**.
* Number of unique decoded byte sequences: **199,998** (no collisions).

**Byte length statistics:**

* Min byte length: **1**
* Max byte length: **128**
* Mean byte length: ≈ **6.99 bytes**
* Distribution by byte length:

  * length = 1: **256 tokens**
  * length 2–4: **49,004 tokens**
  * length 5–8: **99,050 tokens**
  * length 9–16: **47,701 tokens**
  * length >16: **3,987 tokens**

**UTF-8 validity:**

* Tokens whose bytes decode to **ASCII-only** strings: **128,835**
* Tokens whose bytes decode to **valid UTF-8 with non-ASCII chars**: **69,601**
* Tokens whose bytes are **not valid UTF-8**: **1,562**

> Important: those 1,562 tokens are “partial” or otherwise non-UTF-8 sequences; this is **expected** in BPE.
> Your tokenizer must operate on **bytes**, not on Unicode codepoints. You only decode to text when merging all bytes at the very end (or when showing them in the UI as best-effort).

### 3.4 “Single byte” tokens (ranks 0–255)

There are exactly **256 tokens** that decode to byte sequences of length 1.

* These 256 tokens cover all possible byte values **0..255** exactly once:

  ```py
  single_bytes = {decoded[0] for decoded in single_byte_tokens}
  len(single_bytes) == 256
  min(single_bytes) == 0
  max(single_bytes) == 255
  ```

* Their ranks are exactly **0..255** (inclusive):

  * min rank: 0
  * max rank: 255

This is a key property:

> **Any arbitrary byte sequence can always be decomposed into tokens, at worst, as one token per byte**.

This is what makes the BPE algorithm safe: even if no merges apply, you can always fall back to byte-level tokens.

### 3.5 Example: first and last tokens

**First tokens:**

```text
rank 0   "IQ=="  → b'!'  → "!"
rank 1   "Ig=="  → b'"'  → "\""
rank 2   "Iw=="  → b'#'  → "#"
...
rank 15  "MA=="  → b'0'  → "0"
rank 16  "MQ=="  → b'1'  → "1"
...
```

So the very lowest ranks correspond to punctuation and ASCII digits.

**Last few tokens (by rank):**

```text
rank 199988  "CXdz"                    → b"\tw s"         → "\tws" (contains a tab)
rank 199989  "INC60LXQt9C00LXRgQ=="   → 13-byte UTF-8    → some non-ASCII text
rank 199990  "KToo"                   → b"):("           → "):("
rank 199991  "IFByb2R1aXQ="           → b" Produit"      → " Produit"
rank 199992  "QWlyY3JhZnQ="           → b"Aircraft"      → "Aircraft"
rank 199993  "aWZmZW4="               → b"iffen"         → "iffen"
rank 199994  "IHBhdHJvbmVz"           → b" patrones"     → " patrones"
rank 199995  "IHBhcsOibWV0cm9z"       → b" parâmetros"   → " parâmetros"
rank 199996  "Q3Vyc29z"               → b"Cursos"        → "Cursos"
rank 199997  "IGNvY29z"               → b" cocos"        → " cocos"
```

You can see the pattern: higher ranks tend to represent longer and/or rarer substrings/words.

### 3.6 How to use `bpe_ranks` to build the core maps

Inside your worker, after parsing JSON, you should:

#### 3.6.1 Parse and decode the ranks

Pseudo-code (JS):

```js
const parts = bpeRanksStr.split(" ");
if (parts[0] !== "!") {
  throw new Error("Unexpected bpe_ranks format");
}
const offset = Number.parseInt(parts[1], 10);
const b64Tokens = parts.slice(2); // length 199_998

const rankMap = new Map();         // key: "byte0,byte1,...", value: rank
const textMap = new Map();         // key: rank, value: Uint8Array

for (let i = 0; i < b64Tokens.length; i++) {
  const rank = offset + i;
  const b64 = b64Tokens[i];
  const binaryString = atob(b64);
  const bytes = new Uint8Array(binaryString.length);
  for (let j = 0; j < binaryString.length; j++) {
    bytes[j] = binaryString.charCodeAt(j);
  }
  const key = [...bytes].join(",");  // e.g. "104,101,108,108,111"
  rankMap.set(key, rank);
  textMap.set(rank, bytes);
}
```

This reproduces what `js-tiktoken` does (they use `base64-js` instead of `atob`, but same idea).

#### 3.6.2 What `rankMap` and `textMap` are for

* **`rankMap: Map<string, number>`**

  * Keys: comma-separated byte values, e.g. `"104,101,108,108,111"`.
  * Values: integer token IDs (0..199,997).
  * Used by **encoding**:

    * Given a byte sequence (a `Uint8Array`), form its key and check:

      * If key exists: the whole span is a known token.
      * Otherwise: run BPE merges to break it into smaller known spans.

* **`textMap: Map<number, Uint8Array>`**

  * Key: token ID (rank).
  * Value: original bytes sequence representing that token.
  * Used by **decoding**:

    * Given an array of token IDs, map them back to bytes, concatenate, then run `TextDecoder("utf-8")` once at the end.

You should build **both maps** in your worker if you want a full encode+decode implementation or want to debug by showing token bytes explicitly.

---

## 4. How all three fields work together for encoding

Let’s connect the JSON to the algorithm you want in `tiktoken.html`.

### 4.1 Setup (once per loaded JSON)

From the JSON:

1. **Compile the main regex**:

   ```js
   const patStr = data.pat_str;
   const pieceRegex = new RegExp(patStr, "ug");
   ```

2. **Build special token maps and regex**:

   ```js
   const specialTokens = data.special_tokens; // { text: rank }

   const inverseSpecialTokens = {};
   const textEncoder = new TextEncoder();
   for (const [text, rank] of Object.entries(specialTokens)) {
     inverseSpecialTokens[rank] = textEncoder.encode(text);
   }

   const specialRegex = new RegExp(
     Object.keys(specialTokens)
       .map(escapeRegex)
       .join("|"),
     "g"
   );
   ```

3. **Decompress `bpe_ranks` into `rankMap` + `textMap`** (as shown in §3.6).

4. **Create a shared `TextEncoder` and `TextDecoder("utf-8")`** for bytes/text conversions.

### 4.2 Encoding a string with special-token logic

Given:

```ts
encode(
  text: string,
  allowedSpecial: string[] | "all" = [],
  disallowedSpecial: string[] | "all" = "all"
): number[]
```

Algorithm outline:

1. **Build allowed/disallowed sets** from `specialTokens` keys:

   ```js
   const allSpecials = Object.keys(specialTokens);
   const allowedSet = new Set(
     allowedSpecial === "all" ? allSpecials : allowedSpecial
   );

   const disallowedSet = new Set(
     disallowedSpecial === "all"
       ? allSpecials.filter(x => !allowedSet.has(x))
       : disallowedSpecial
   );
   ```

2. **If any specials are disallowed, validate the input text**:

   * Build `disallowedRegex` from `disallowedSet`.
   * If `text.match(disallowedRegex)` is non-null, throw:

     ```js
     throw new Error(
       "The text contains a special token that is not allowed: " + match[0]
     );
     ```

3. **Walk the string, splitting around allowed specials**:

   Pseudocode:

   ```js
   const tokens: number[] = [];
   let start = 0;

   while (true) {
     // Find next allowed special
     let nextSpecial = null;
     let searchPos = start;

     while (true) {
       specialRegex.lastIndex = searchPos;
       const m = specialRegex.exec(text);
       if (!m || allowedSet.has(m[0])) {
         nextSpecial = m;
         break;
       }
       searchPos = m.index + 1;
     }

     const end = nextSpecial ? nextSpecial.index : text.length;

     // Encode segment [start, end) with pieceRegex + BPE:
     const segment = text.substring(start, end);
     // For each match of pieceRegex on segment, run BPE encoding and push token IDs.
     // (see §4.3 below)

     if (!nextSpecial) break;

     // Handle the special token itself:
     const specialStr = nextSpecial[0];
     tokens.push(specialTokens[specialStr]);

     start = nextSpecial.index + specialStr.length;
   }

   return tokens;
   ```

**Important:** This special handling logic is **not in the JSON**, but the JSON tells you which strings are special and what IDs they map to. Your implementation must layer the algorithm above on top of `special_tokens`.

### 4.3 Encoding each “piece” with BPE and the rank maps

For each piece matched by `pieceRegex` (e.g. `"hello"`, `" world"`, emojis, etc.):

1. Get bytes:

   ```js
   const bytes = textEncoder.encode(piece); // Uint8Array
   ```

2. Try to see if the entire piece is a known token:

   ```js
   const key = [...bytes].join(",");
   const directRank = rankMap.get(key);
   if (directRank !== undefined) {
     tokens.push(directRank);
     continue;
   }
   ```

3. Otherwise, run the **byte pair merge** algorithm using `rankMap`:

   * You implement `bytePairMerge(piece: Uint8Array, ranks: Map<string, number>)`
     that repeatedly merges adjacent pairs with the **lowest rank** until no more merges are found.
   * Then `bytePairEncode` maps those merged spans back to ranks:

     ```ts
     function bytePairEncode(piece: Uint8Array, ranks: Map<string, number>): number[] {
       if (piece.length === 1) return [ranks.get([...piece].join(","))!];

       return bytePairMerge(piece, ranks)
         .map(p => ranks.get(piece.slice(p.start, p.end).join(",")))
         .filter((x): x is number => x != null);
     }
     ```

4. Append the returned ranks for that piece to the overall token stream.

**How `bpe_ranks` is used here**:

* `rankMap` is exactly what guides **which merges are allowed** and what their scores (ranks) are.
* The fact `0..255` correspond to single bytes ensures that, if no merges are possible, `bytePairEncode` can always fall back to 1-byte tokens.

---

## 5. Decoding: using `textMap` and `inverseSpecialTokens`

To reconstruct text:

1. Given an array of token IDs, for each token:

   * If `textMap.has(id)`, use that bytes sequence.
   * Else if `inverseSpecialTokens[id]` exists, use that bytes sequence.
   * Else, drop or error (should not happen if IDs come from this encoding).

2. Concatenate all `Uint8Array`s into a single big `Uint8Array`.

3. Run `new TextDecoder("utf-8").decode(mergedBytes)`.

This uses:

* `textMap` that you derived from `bpe_ranks`.
* `inverseSpecialTokens` that you derived from `special_tokens`.

---

## 6. What does *not* exist in the JSON (so don’t rely on it)

The JSON is deliberately minimal. Things that **are not** in `o200k_base.json`:

* No explicit **vocabulary size** field – but from counts:

  * 199,998 BPE tokens + 2 specials = 200,000 used IDs.
* No **model name → encoding** mapping – that mapping lives in higher-level libraries (e.g. the `getEncodingNameForModel` switch).
* No **frequency** or **probability** metadata per token – you only get ranks, not how common tokens are.
* No **human-readable token strings** – you must base64-decode and then (optionally) UTF-8 decode; some tokens are not valid UTF-8.
* No **BPE rules** in human-readable form – all merges are encoded implicitly through the ranks table, not as “pair → new token” lists.

So when rewriting `tiktoken.html`:

* You **must not** try to infer anything beyond:

  * “This byte sequence has this rank”
  * “This special string has this rank”
  * “These pieces are identified by `pat_str`.”
* All other behavior (allowed/disallowed specials, model selection, etc.) should be implemented in your JS logic on top of this data.

---

## 7. TL;DR: Checklist for a correct `tiktoken.html`

If you want to be able to rewrite your HTML/worker from this document, here’s a practical checklist tied back to the JSON:

1. **On load (`init`) in the worker:**

   * Parse the JSON.
   * Store `pat_str` and build `pieceRegex = new RegExp(pat_str, "ug")`.
   * Store `special_tokens` into `specialTokens`.
   * Build:

     * `inverseSpecialTokens` via `TextEncoder`.
     * `specialRegex` from the keys of `specialTokens`.
   * Parse and decode `bpe_ranks`:

     * Ensure `parts[0] === "!"`.
     * `offset = parseInt(parts[1], 10)`.
     * `tokens = parts.slice(2)`.
     * Build `rankMap` (key = comma-joined bytes) and `textMap` (rank → bytes).

2. **For each encode request:**

   * Construct `allowedSet` and `disallowedSet` from the keys of `specialTokens`.
   * If disallowed is non-empty, scan and throw on any disallowed special.
   * Walk text, slicing around **allowed** special tokens:

     * For plain segments: apply `pieceRegex` and BPE logic (using `rankMap`).
     * For special segments: map via `specialTokens` and push the ID.

3. **For any decode request:**

   * For each token ID:

     * If in `textMap`, use those bytes.
     * Else if in `inverseSpecialTokens`, use those bytes.
   * Concatenate bytes and decode UTF-8 once.

If you follow this mapping from JSON → data structures → encode/decode behavior, your `tiktoken.html` implementation will be aligned with how the professionals’ `js-tiktoken` uses `o200k_base.json`, and you’ll avoid the subtle edge cases (special tokens, invalid UTF-8, single-byte fallback, etc.) that tend to bite hand-rolled implementations.
