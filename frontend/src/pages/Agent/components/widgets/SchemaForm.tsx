import type { ReactNode } from 'react';
import { iCls, sCls, FLabel } from '../common/form';

/**
 * A lightweight JSON-Schema → form renderer for a model's bespoke ``model_params``
 * (the JSON Schema of its ``CONFIG_SCHEMA``, carried in the garden manifest as
 * ``config_schema``). Handles the scalar field shapes Pydantic emits
 * (number / integer / boolean / string / enum) and falls back to a JSON text input
 * for arrays / objects (e.g. a CFA ``factor_assignment`` list). Read-only when not
 * ``editable``.
 */

export interface JsonSchemaProp {
  type?: string;
  default?: unknown;
  title?: string;
  description?: string;
  enum?: (string | number)[];
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  anyOf?: JsonSchemaProp[];
  items?: JsonSchemaProp;
}

export interface JsonSchema {
  title?: string;
  properties?: Record<string, JsonSchemaProp>;
  required?: string[];
}

/** The effective scalar type, resolving Optional (anyOf [T, null]) to T. */
function resolveType(prop: JsonSchemaProp): string {
  if (prop.type) return prop.type;
  const branch = (prop.anyOf || []).find((b) => b.type && b.type !== 'null');
  return branch?.type ?? 'string';
}

export function SchemaForm({
  schema,
  values,
  editable,
  onChange,
}: {
  schema: JsonSchema;
  values: Record<string, unknown>;
  editable: boolean;
  onChange: (next: Record<string, unknown>) => void;
}) {
  const props = schema.properties || {};
  const keys = Object.keys(props);
  if (keys.length === 0) {
    return <p className="text-xs text-ink-400">This model declares no configurable parameters.</p>;
  }

  const set = (key: string, value: unknown) => onChange({ ...values, [key]: value });

  return (
    <div className="space-y-3">
      {keys.map((key) => {
        const prop = props[key];
        const label = prop.title || key;
        const t = resolveType(prop);
        const raw = key in values ? values[key] : prop.default;
        const min = prop.exclusiveMinimum ?? prop.minimum;
        const max = prop.exclusiveMaximum ?? prop.maximum;

        let field: ReactNode;
        if (prop.enum && prop.enum.length > 0) {
          field = (
            <select
              className={sCls}
              disabled={!editable}
              value={String(raw ?? '')}
              onChange={(e) => set(key, e.target.value)}
            >
              {prop.enum.map((o) => (
                <option key={String(o)} value={String(o)}>
                  {String(o)}
                </option>
              ))}
            </select>
          );
        } else if (t === 'number' || t === 'integer') {
          field = (
            <input
              type="number"
              className={iCls}
              disabled={!editable}
              step={t === 'integer' ? 1 : 'any'}
              min={min}
              max={max}
              value={raw === undefined || raw === null ? '' : Number(raw)}
              onChange={(e) => {
                const v = e.target.value;
                set(key, v === '' ? null : t === 'integer' ? parseInt(v, 10) : parseFloat(v));
              }}
            />
          );
        } else if (t === 'boolean') {
          field = (
            <select
              className={sCls}
              disabled={!editable}
              value={raw ? 'true' : 'false'}
              onChange={(e) => set(key, e.target.value === 'true')}
            >
              <option value="true">true</option>
              <option value="false">false</option>
            </select>
          );
        } else if (t === 'array' || t === 'object') {
          // Best-effort JSON editor for compound params (e.g. factor_assignment).
          field = (
            <input
              type="text"
              className={`${iCls} font-mono`}
              disabled={!editable}
              defaultValue={raw === undefined || raw === null ? '' : JSON.stringify(raw)}
              placeholder={t === 'array' ? '[0, 0, 1, 1]' : '{ }'}
              onBlur={(e) => {
                const v = e.target.value.trim();
                if (v === '') return set(key, null);
                try {
                  set(key, JSON.parse(v));
                } catch {
                  /* leave the previous value; invalid JSON is ignored */
                }
              }}
            />
          );
        } else {
          field = (
            <input
              type="text"
              className={iCls}
              disabled={!editable}
              value={raw === undefined || raw === null ? '' : String(raw)}
              onChange={(e) => set(key, e.target.value)}
            />
          );
        }

        return (
          <div key={key}>
            <FLabel>{label}</FLabel>
            {field}
            {prop.description && (
              <p className="text-[10px] text-ink-300 mt-0.5 leading-snug">{prop.description}</p>
            )}
          </div>
        );
      })}
    </div>
  );
}
