// api/recommend.js
// Vercel Serverless Function — Nestly → OpenAI Responses API proxy
// - Reads OPENAI_API_KEY from env (never commit secrets)
// - Validates request shape and size
// - Uses OpenAI Responses API with Structured Outputs (JSON Schema)
// - Validates model output with AJV as a second layer
// - Restricts CORS to ALLOWED_ORIGIN

import Ajv from "ajv";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || ""; // e.g. https://allisonsaretsky.github.io
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-5.4-mini";
const TIMEOUT_MS = parseInt(process.env.TIMEOUT_MS || "15000", 10);
const MAX_CONTENT_LENGTH = parseInt(process.env.MAX_CONTENT_LENGTH || "200000", 10); // ~200KB

function sendJson(res, status, body) {
  res.status(status).json(body);
}

function basicRequestValidate(body) {
  if (!body || typeof body !== "object") return "body-must-be-json-object";
  if (!body.run_id || typeof body.run_id !== "string") return "missing_or_invalid_run_id";
  if (!body.family || typeof body.family !== "object") return "missing_family_object";
  if (!Array.isArray(body.care_hierarchy)) return "missing_care_hierarchy_array";
  if (!Array.isArray(body.providers) || body.providers.length === 0) return "missing_providers_array";
  // providers must include provider_id strings
  for (const p of body.providers) {
    if (!p || typeof p !== "object") return "invalid_provider_record";
    if (!p.provider_id || typeof p.provider_id !== "string") return "provider_missing_id";
  }
  if (!Array.isArray(body.matching_priorities)) return "missing_matching_priorities";
  return null;
}

// Nestly structured output JSON Schema (draft-07)
const nestlySchema = {
  $schema: "http://json-schema.org/draft-07/schema#",
  type: "object",
  required: [
    "care_type",
    "ranked_providers",
    "excluded_providers",
    "reasoning",
    "confidence",
    "missing_information",
    "requires_caregiver_review"
  ],
  additionalProperties: false,
  properties: {
    care_type: { type: "string" },
    ranked_providers: {
      type: "array",
      items: {
        type: "object",
        required: ["provider_id", "provider_name", "rank", "reason"],
        additionalProperties: false,
        properties: {
          provider_id: { type: "string" },
          provider_name: { type: "string" },
          rank: { type: "integer", minimum: 1 },
          reason: { type: "string" }
        }
      }
    },
    excluded_providers: {
      type: "array",
      items: {
        type: "object",
        required: ["provider_id", "provider_name", "reason"],
        additionalProperties: false,
        properties: {
          provider_id: { type: "string" },
          provider_name: { type: "string" },
          reason: { type: "string" }
        }
      }
    },
    reasoning: { type: "string" },
    confidence: { type: "string", enum: ["High", "Medium", "Low"] },
    missing_information: { type: "array", items: { type: "string" } },
    requires_caregiver_review: { type: "boolean" }
  }
};

const ajv = new Ajv({ allErrors: true, strict: false });
const validateModel = ajv.compile(nestlySchema);

export default async function handler(req, res) {
  const origin = req.headers.origin || "";

  // CORS preflight
  if (req.method === "OPTIONS") {
    const allowOrigin = ALLOWED_ORIGIN || origin || "*";
    res.setHeader("Access-Control-Allow-Origin", allowOrigin);
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    res.status(204).end();
    return;
  }

  const allowOrigin = ALLOWED_ORIGIN || origin || "*";
  res.setHeader("Access-Control-Allow-Origin", allowOrigin);
  res.setHeader("Vary", "Origin");

  if (req.method !== "POST") {
    return sendJson(res, 405, { error: "method_not_allowed" });
  }

  const contentLength = parseInt(req.headers["content-length"] || "0", 10);
  if (contentLength > MAX_CONTENT_LENGTH) {
    console.warn("recommend: payload_too_large", { origin, payloadSize: contentLength });
    return sendJson(res, 413, { error: "payload_too_large" });
  }

  let body = req.body;
  // If body is string, try parse
  if (typeof body === "string") {
    try { body = JSON.parse(body); } catch (e) { /* fallthrough */ }
  }

  const validationError = basicRequestValidate(body);
  if (validationError) {
    console.warn("recommend: invalid_request", { validationError, origin });
    return sendJson(res, 400, { error: "invalid_request", detail: validationError });
  }

  if (!OPENAI_API_KEY) {
    console.error("recommend: missing_openai_api_key");
    return sendJson(res, 500, { error: "server_misconfiguration" });
  }

  // Build Responses API structured outputs payload using text.format with json_schema
  const responseSchema = nestlySchema; // reuse nestly schema as structured output

  // Prepare model messages / context (concise)
  const systemInstruction = `You are Nestly, a childcare recommendation assistant. Use the provided family, providers, policies, and matching priorities to return a single JSON object matching the provided schema exactly. Do not include extra fields, explanations, or actions. Do not invent caregiver consent or eligibility. If information is missing, list exact missing fields in 'missing_information'. IMPORTANT: Always set the field "requires_caregiver_review" to true (boolean) in the returned JSON. This is a fixed product boundary and must not be returned as false by the model. The server will also enforce this deterministically as a backstop.`;

  const openaiRequest = {
    model: OPENAI_MODEL,
    input: [
      { role: "system", content: systemInstruction },
      { role: "user", content: JSON.stringify({ family: body.family, care_hierarchy: body.care_hierarchy, providers: body.providers, matching_priorities: body.matching_priorities }) }
    ],
    // Use Responses API Structured Outputs via text.format with schema
    text: {
      format: {
        type: "json_schema",
        name: "nestly_recommendation",
        strict: true,
        schema: responseSchema
      }
    },
    // Do not store model responses on provider side
    store: false
  };

  // Call OpenAI Responses API
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    // Minimal/no-PII logging
    console.info("recommend: invoke_model", { run_id: body.run_id, origin });

    const resp = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify(openaiRequest),
      signal: controller.signal
    });
    clearTimeout(timeout);

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      console.warn("recommend: upstream_error", { status: resp.status, origin });
      return sendJson(res, 502, { error: "upstream_model_error", status: resp.status, detail: text.slice(0, 1000) });
    }

    const json = await resp.json();

    // Extract assistant textual output (output_text) and parse it as JSON
    let modelOutput = null;

    // Preferred: some Responses responses include `output_text` at top-level
    if (json.output_text && typeof json.output_text === "string") {
      try {
        modelOutput = JSON.parse(json.output_text);
      } catch (e) {
        // continue to other extraction methods
      }
    }

    // Fallback: inspect output array for content items with type 'output_text' and assemble text
    if (!modelOutput && Array.isArray(json.output)) {
      let combined = "";
      for (const out of json.output) {
        if (out && Array.isArray(out.content)) {
          for (const c of out.content) {
            if ((c.type === "output_text" || c.type === "text" || c.type === "message") && (typeof c.text === "string" || typeof c.value === "string")) {
              combined += (c.text || c.value || "") + "\n";
            }
          }
        }
      }
      if (combined) {
        try {
          modelOutput = JSON.parse(combined);
        } catch (e) {
          // ignore parse error
        }
      }
    }

    // Another fallback: output_parsed if provider populated it
    if (!modelOutput && json.output_parsed) {
      modelOutput = json.output_parsed;
    }

    if (!modelOutput) {
      console.warn("recommend: model_no_structured_output", { run_id: body.run_id });
      return sendJson(res, 502, { error: "model_no_structured_output" });
    }

    // Server-side AJV validation (second-layer)
    const valid = validateModel(modelOutput);
    if (!valid) {
      console.warn("recommend: model_schema_invalid", { run_id: body.run_id, origin });
      return sendJson(res, 502, { error: "model_output_schema_invalid", detail: validateModel.errors });
    }

    // Verify that provider_ids in modelOutput belong to provided context
    const allowedIds = new Set(body.providers.map((p) => p.provider_id));
    const invalidProviders = [];
    for (const p of modelOutput.ranked_providers || []) {
      if (!allowedIds.has(p.provider_id)) invalidProviders.push(p.provider_id);
    }
    for (const e of modelOutput.excluded_providers || []) {
      if (!allowedIds.has(e.provider_id)) invalidProviders.push(e.provider_id);
    }
    if (invalidProviders.length > 0) {
      console.warn("recommend: model_referenced_unknown_provider", { run_id: body.run_id, origin });
      return sendJson(res, 502, { error: "model_referenced_unknown_provider", detail: [...new Set(invalidProviders)] });
    }

    // Deterministic safety overrides (server-side protective layer):
    // 1) Always require caregiver review (cannot be weakened by model)
    // 2) Suppress unnecessary missing-information requests (exact/overly-specific asks)
    const overrides = {};
    if (modelOutput.requires_caregiver_review !== true) {
      overrides.requires_caregiver_review = { before: modelOutput.requires_caregiver_review, after: true };
      modelOutput.requires_caregiver_review = true;
    }

    // Filter missing_information conservatively:
    // - Remove non-essential requests that ask for "exact" values or general precision (exact dates/times).
    // - Remove child-age requests unless a provider has an explicit hard age eligibility requirement that cannot be evaluated without age.
    // - Remove requests to confirm recurring vs. occasional overnight coverage (provider capacity/confirmation should remain in provider reason text).
    const removedMissing = [];
    if (Array.isArray(modelOutput.missing_information)) {
      const filtered = [];

      // detect if any provider requires child age as a hard eligibility rule
      let ageRequiredByProvider = false;
      for (const p of (body.providers || [])) {
        // explicit eligibility_requirements keys like min_age_months or min_age
        const reqs = p.eligibility_requirements || {};
        for (const k of Object.keys(reqs)) {
          if (/min_age|min_age_months|minimum_age|minimum_age_months/i.test(k)) {
            ageRequiredByProvider = true;
            break;
          }
        }
        if (ageRequiredByProvider) break;
        // also inspect provider policies for hard minimum-age constraints
        if (Array.isArray(p.policies)) {
          for (const pol of p.policies) {
            const t = (pol.policy_type || "") + " " + (pol.enforcement || "");
            if (/minimum age|min age|minimum_age/i.test(String(pol.policy_type || pol.policy_rule || "")) && /hard constraint|hard/i.test(String(pol.enforcement || pol.policy_rule || ""))) {
              ageRequiredByProvider = true;
              break;
            }
          }
        }
      }

      for (const item of modelOutput.missing_information) {
        if (!item || typeof item !== "string") continue;
        const lower = item.toLowerCase();

        // Patterns considered non-essential for safe recommendation
        const nonEssentialExact = /\bexact\b|\bexactly\b|\bexact start date\b|\bexact start time\b|\bexact .* time(s)?\b/i;
        const agePattern = /child('?s)? age|child age|age of the child|date of birth|dob|due date/i;
        const overnightPattern = /overnight frequency|overnight dates?|overnight times?|recurring overnight|occasional overnight|overnight frequency/i;

        // Drop non-essential exactness requests
        if (nonEssentialExact.test(lower)) {
          removedMissing.push(item);
          continue;
        }

        // Drop overnight frequency requests (provider-specific capacity/confirmation should remain in provider reason)
        if (overnightPattern.test(lower)) {
          removedMissing.push(item);
          continue;
        }

        // Keep child age only if some provider requires it as a hard eligibility check
        if (agePattern.test(lower)) {
          if (ageRequiredByProvider) {
            filtered.push(item);
          } else {
            removedMissing.push(item);
          }
          continue;
        }

        // Otherwise keep the item
        filtered.push(item);
      }

      modelOutput.missing_information = filtered;
      if (removedMissing.length > 0) overrides.removed_missing_information = (overrides.removed_missing_information || []).concat(removedMissing);
    }

    // Enforce hard-family-constraint precedence over default hierarchy.
    // Detect strong/ hard constraints from family rules or careNeed/schedule indicating overnight/extended needs.
    const familyText = ((body.family && ((body.family.rules || []).join(' ')) ) || '') + ' ' + (body.family && (body.family.schedule || body.family.careNeed || ''));
    const needsExtended = /overnight|overnight coverage|overnight-call|overnight call|rotating|extended-?hours|24 hour|24-hour/i.test(familyText);

    if (needsExtended) {
      // Identify providers that can meet extended/overnight needs
      const capable = (body.providers || []).filter((p) => {
        const hours = String(p.hours || p.operating_hours || '').toLowerCase();
        const providerType = String(p.provider_type || p.mapped_childcare_type || p.care_type || '').toLowerCase();
        if (/24\s*hour|24hr|24-?hours|overnight|flexible|overnight-call|overnight call/.test(hours)) return true;
        if (/in-?home|nanny|home caregiver|in home caregiver/.test(providerType)) return true;
        return false;
      }).map(p => p.provider_id);

      if (capable.length > 0) {
        // Allowed care types based on capable providers
        const allowedTypes = new Set();
        for (const p of (body.providers || [])) {
          if (capable.includes(p.provider_id)) {
            const t = p.mapped_childcare_type || p.provider_type || p.care_type || null;
            if (t) allowedTypes.add(String(t));
          }
        }

        const modelCareType = modelOutput.care_type;
        const modelCareTypeInAllowed = allowedTypes.has(modelCareType);

        // If In-Home providers exist that can meet extended needs, prefer In-Home care_type
        const inHomeAvailable = (body.providers || []).some(p => {
          const t = String(p.provider_type || p.mapped_childcare_type || p.care_type || '').toLowerCase();
          return /in-?home|nanny|home caregiver/.test(t) || /flexible|overnight/.test(String(p.hours || p.operating_hours || '').toLowerCase());
        });

        const familyPref = (body.family && (body.family.explicit_care_type_preferences || body.family.recommendation)) || null;

        let replacement = null;

        if (inHomeAvailable && (!familyPref || (typeof familyPref === 'string' ? /in-?home/i.test(String(familyPref)) : (Array.isArray(familyPref) ? familyPref.some(fp => /in-?home/i.test(String(fp))) : false)))) {
          replacement = 'In-Home Caregiver';
        } else if (!modelCareTypeInAllowed) {
          // Pick a replacement care type from capable providers
          for (const p of (body.providers || [])) {
            if (!capable.includes(p.provider_id)) continue;
            const t = p.mapped_childcare_type || p.provider_type || p.care_type || null;
            if (!t) continue;
            const tstr = String(t);
            if (!replacement) replacement = tstr;
          }
          if (!replacement) replacement = modelCareType; // fallback
        }

        if (replacement && replacement !== modelCareType) {
          overrides.hard_constraint_override = { before: modelCareType, after: replacement };
          modelOutput.care_type = replacement;

          // Reorder ranked_providers to favor capable providers and match replacement type first
          if (Array.isArray(modelOutput.ranked_providers) && modelOutput.ranked_providers.length > 0) {
            const capSet = new Set(capable);
            const replLower = String(replacement).toLowerCase();
            modelOutput.ranked_providers.sort((a,b) => {
              const aRepl = String(a.provider_type || a.mapped_childcare_type || '').toLowerCase().includes(replLower) ? 0 : 1;
              const bRepl = String(b.provider_type || b.mapped_childcare_type || '').toLowerCase().includes(replLower) ? 0 : 1;
              const aCap = capSet.has(a.provider_id) ? 0 : 1;
              const bCap = capSet.has(b.provider_id) ? 0 : 1;
              return (aRepl - bRepl) || (aCap - bCap);
            });
          }
        }
      }
    }

    const serverVerification = { ok: true, timestamp: new Date().toISOString() };
    if (Object.keys(overrides).length > 0) serverVerification.overrides = overrides;

    // Success: Return only the validated structured model output (no raw model trace)
    return sendJson(res, 200, { model_output: modelOutput, server_verification: serverVerification });
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === "AbortError") {
      console.warn("recommend: model_timeout", { run_id: (req.body && req.body.run_id) || null, origin });
      return sendJson(res, 504, { error: "upstream_timeout" });
    }
    console.error("recommend: unexpected_error", { err: (err && err.message) || String(err) });
    return sendJson(res, 500, { error: "internal_server_error" });
  }
}
