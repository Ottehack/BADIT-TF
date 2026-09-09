#!/usr/bin/env python3
"""Render the TPAMI-schema workbook into one auditable Markdown deliverable."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from openpyxl import load_workbook
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cell(v):
 if v is None:return ""
 return str(v).replace("|","\\|").replace("\n","<br>")
def main():
 p=argparse.ArgumentParser();p.add_argument('--workbook',type=Path,default=ROOT/'experiments/results/BADIT_TF_TPAMI_EMPTY_TABLES.xlsx');p.add_argument('--output',type=Path,default=ROOT/'BADIT_TF_ALL_RESULTS.md');a=p.parse_args()
 w=load_workbook(a.workbook,read_only=True,data_only=False);lines=['# BADIT-TF All Results','',f'Generated: {datetime.now(timezone.utc).isoformat()}',f'Source workbook: `{a.workbook.relative_to(ROOT)}`',f'Source SHA256: `{sha(a.workbook)}`','','缺失值严格保留为 `[RESULT NEEDED]`；所有数值来自当前实验，不借用论文旧数字。','']
 for s in w:
  if not s.title.startswith('Table_'):continue
  lines += [f"## {s.title.replace('_',' ')}",'',cell(s['A2'].value),'']
  headers=[cell(c.value) for c in s[4] if c.value is not None]
  lines += ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
  for row in s.iter_rows(min_row=5,max_col=len(headers),values_only=True):
   if not any(v is not None for v in row):continue
   lines.append('| '+' | '.join(cell(v) for v in row)+' |')
  lines.append('')
 w.close();a.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 manifest={'schema_version':1,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'source_workbook':str(a.workbook.relative_to(ROOT)),'source_sha256':sha(a.workbook),'output':str(a.output.relative_to(ROOT)),'output_sha256':sha(a.output),'missing_marker':'[RESULT NEEDED]'}
 mp=a.output.with_suffix('.manifest.json');mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps(manifest,sort_keys=True))
if __name__=='__main__':main()
