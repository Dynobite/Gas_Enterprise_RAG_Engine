"""
Automated Graphify Knowledge Graph Synchronization Engine.
Handles automatic incremental extraction of knowledge ontologies when documents are uploaded or deleted.
"""

import os
import sys
import shutil
import subprocess
import threading
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("graph_sync")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Add console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    
    # Add file handler in project root
    log_file = os.path.join(project_root, "graph_sync.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

# Global lock to ensure only one Graphify extraction runs at a time
_GRAPH_SYNC_LOCK = threading.Lock()

def get_graph_dirs():
    docs_kg_dir = os.path.join(project_root, "docs_knowledge_graph")
    graph_out_dir = os.path.join(project_root, "graph_knowledge")
    os.makedirs(docs_kg_dir, exist_ok=True)
    os.makedirs(graph_out_dir, exist_ok=True)
    return docs_kg_dir, graph_out_dir

def sync_knowledge_graph_task(added_filename: str = None, deleted_filename: str = None):
    """
    Background worker executed after document ingestion or deletion.
    Acquires thread lock so concurrent uploads queue sequentially.
    """
    if not _GRAPH_SYNC_LOCK.acquire(blocking=False):
        logger.info("[GraphSync] Another graph extraction is already running. Waiting for lock...")
        _GRAPH_SYNC_LOCK.acquire()

    try:
        docs_kg_dir, graph_out_dir = get_graph_dirs()
        markdown_dir = os.path.join(project_root, "data", "processed", "markdown")

        # 1. Handle Added Document
        if added_filename:
            logger.info(f"[GraphSync] Adding document to knowledge graph: {added_filename}")
            candidate_mds = [
                os.path.join(markdown_dir, f"{added_filename}.md"),
                os.path.join(markdown_dir, added_filename if added_filename.endswith(".md") else f"{added_filename}.md")
            ]
            
            src_md = None
            for cand in candidate_mds:
                if os.path.exists(cand):
                    src_md = cand
                    break

            if not src_md and os.path.exists(markdown_dir):
                for f in os.listdir(markdown_dir):
                    if f.lower().startswith(added_filename.lower()) and f.endswith(".md"):
                        src_md = os.path.join(markdown_dir, f)
                        break

            if src_md and os.path.exists(src_md):
                base_name = os.path.basename(src_md)
                clean_name = base_name.replace(" ", "_").replace("(", "_").replace(")", "_")
                dest_md = os.path.join(docs_kg_dir, clean_name)

                # Cap max preview to 45 KB for fast, timeout-safe LLM extraction
                max_bytes = 45 * 1024
                file_size = os.path.getsize(src_md)
                if file_size > max_bytes:
                    with open(src_md, "r", encoding="utf-8", errors="replace") as fin:
                        head_content = fin.read(max_bytes)
                    with open(dest_md, "w", encoding="utf-8") as fout:
                        fout.write(head_content + "\n\n<!-- Truncated for ontology extraction -->\n")
                    logger.info(f"[GraphSync] Capped {clean_name} to 45KB for fast extraction.")
                else:
                    shutil.copy2(src_md, dest_md)
                    logger.info(f"[GraphSync] Copied full {clean_name} ({file_size} bytes).")

        # 2. Handle Deleted Document
        if deleted_filename:
            logger.info(f"[GraphSync] Removing document from knowledge graph: {deleted_filename}")
            base_del = deleted_filename.replace(" ", "_").replace("(", "_").replace(")", "_").lower()
            if os.path.exists(docs_kg_dir):
                for f in os.listdir(docs_kg_dir):
                    f_clean = f.lower()
                    if base_del in f_clean or f_clean.startswith(base_del):
                        target_del = os.path.join(docs_kg_dir, f)
                        try:
                            os.remove(target_del)
                            logger.info(f"[GraphSync] Deleted {target_del}")
                        except Exception as e:
                            logger.warning(f"[GraphSync] Could not delete {target_del}: {e}")

        # 3. Locate graphify binary
        graphify_bin = shutil.which("graphify")
        if not graphify_bin:
            candidates = [
                os.path.expanduser("~/.local/bin/graphify"),
                "/home/ollamauser/.local/bin/graphify",
                "/home/user/.local/bin/graphify"
            ]
            for c in candidates:
                if os.path.exists(c):
                    graphify_bin = c
                    break

        if not graphify_bin:
            logger.error("[GraphSync] graphify binary not found on PATH.")
            return

        # 4. Run incremental extraction
        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "http://192.168.152.74:11434/v1")
        env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "ollama")

        extract_cmd = [
            graphify_bin,
            "extract",
            docs_kg_dir,
            "--backend", "openai",
            "--model", "gpt-fast",
            "--max-concurrency", "1",
            "--api-timeout", "240",
            "--out", graph_out_dir
        ]

        logger.info(f"[GraphSync] Executing extraction: {docs_kg_dir}")
        res_extract = subprocess.run(extract_cmd, env=env, capture_output=True, text=True)
        if res_extract.returncode != 0:
            logger.warning(f"[GraphSync] Extraction stderr: {res_extract.stderr}")
        else:
            logger.info("[GraphSync] Incremental extraction completed successfully.")

        # 5. Enrich graph.json and export interactive HTML
        graph_json = os.path.join(graph_out_dir, "graphify-out", "graph.json")
        graph_html = os.path.join(graph_out_dir, "graphify-out", "graph.html")
        if os.path.exists(graph_json):
            _enrich_graph_json(graph_json)
            export_cmd = [
                graphify_bin,
                "export",
                "html",
                "--graph", graph_json
            ]
            res_export = subprocess.run(export_cmd, env=env, capture_output=True, text=True)
            logger.info(f"[GraphSync] Export HTML status: {res_export.returncode}")
            
            if os.path.exists(graph_html):
                _enrich_graph_html(graph_html)

        # 6. If on Machine 1, sync to Machine 2 over LAN if reachable
        _sync_to_peer_if_applicable(graph_out_dir)

    except Exception as e:
        logger.error(f"[GraphSync] Error during graph synchronization: {e}", exc_info=True)
    finally:
        _GRAPH_SYNC_LOCK.release()


def _sync_to_peer_if_applicable(graph_out_dir: str):
    """Syncs newly generated graph files to peer node over local network if paramiko is available."""
    try:
        import socket
        hostname = socket.gethostname()
        if "c14753" in hostname:
            peer_ip = "192.168.152.74"
            peer_user = "user"
        elif "c11171" in hostname:
            peer_ip = "192.168.152.38"
            peer_user = "ollamauser"
        else:
            return

        peer_password = os.getenv("PEER_SSH_PASSWORD", "")
        if not peer_password:
            logger.debug("[GraphSync] No PEER_SSH_PASSWORD environment variable set; skipping remote peer sync.")
            return

        import paramiko
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(peer_ip, username=peer_user, password=peer_password, timeout=4)
        
        sftp = client.open_sftp()
        local_html = os.path.join(graph_out_dir, "graphify-out", "graph.html")
        local_json = os.path.join(graph_out_dir, "graphify-out", "graph.json")

        peer_target_dir = f"/home/{peer_user}/RAG_SYSTEM/graph_knowledge/graphify-out"
        client.exec_command(f"mkdir -p '{peer_target_dir}'")

        if os.path.exists(local_html):
            sftp.put(local_html, f"{peer_target_dir}/graph.html")
        if os.path.exists(local_json):
            sftp.put(local_json, f"{peer_target_dir}/graph.json")

        sftp.close()
        client.close()
        logger.info(f"[GraphSync] Successfully synced graph to peer {peer_ip}.")
    except Exception as e:
        logger.debug(f"[GraphSync] Peer sync skipped: {e}")


STANDARDS_MAP = {
    "gost_9544_2015_standard": "ГОСТ 9544-2015 (Нормы герметичности затворов)",
    "gost_28343_89_standard": "ГОСТ 28343-89 (Краны шаровые стальные фланцевые)",
    "gost_6111_52_standard": "ГОСТ 6111-52 (Резьба коническая дюймовая)",
    "iso_5211_standard": "ISO 5211 (Присоединительные фланцы приводов)",
    "gost_12815_flanges_gost_12815_80": "ГОСТ 12815-80 (Фланцы арматуры и трубопроводов)",
    "gost_12815_flanges_sto_gazprom": "СТО Газпром (Требования к обвязке арматуры)",
    "gost_12815_flanges_document": "ГОСТ 12815-80 (Фланцы арматуры, соединительных частей)",
    "gost_28343_ball_valves_summary_file": "ГОСТ 28343-89 (Сводные требования к кранам шаровым)",
    "bom_butterfly_valve_dn80_bom": "[BOM] Спецификация деталей затвора DN80",
    "dfmea_butterfly_valve_dfmea": "[DFMEA] Матрица рисков и отказов затвора DN80",
    "defect_act_dn80_sealing_defect_act": "[Акт] Дефектовка: Протечка седлового уплотнения DN80",
    "valve_dn80": "Затвор дисковый поворотный DN80",
    "part_dpm_a_491425_005": "[BOM DN80] Затвор дисковый в сборе (ДПМ.А.491425.005)",
    "part_dpm_a_301116_005": "[BOM DN80] Корпус затвора (ДПМ.А.301116.005)",
    "part_dpm_a_301241_012": "[BOM DN80] Диск затвора (ДПМ.А.301241.012)",
    "part_dpm_a_301511_008": "[BOM DN80] Шток затвора (ДПМ.А.301511.008)",
    "part_dpm_a_302612_010": "[BOM DN80] Седловое уплотнение PTFE (ДПМ.А.302612.010)",
    "steel_flanged_ball_valve": "Кран шаровой стальной фланцевый",
    "natural_gas_pipeline_application": "Область: Газопроводы природного газа",
    "pn_16_to_100_pressure_range": "Параметр: Давление PN 16 — PN 100",
    "steel_09g2c_material": "Материал: Сталь 09Г2С (хладостойкая)",
    "steel_20h13_material": "Материал: Сталь 20Х13 (коррозионностойкая)",
    "gost_12815_flanges_stal_12x18n10t": "Материал: Сталь 12Х18Н10Т (нержавеющая)",
    "gost_12815_flanges_stal_20": "Материал: Сталь 20 (углеродистая)",
    "gost_12815_flanges_ispolnenie_1": "[ГОСТ 12815] Исполнение 1 (с соединительным выступом)",
    "gost_12815_flanges_ispolnenie_2": "[ГОСТ 12815] Исполнение 2 (с выступом)",
    "gost_12815_flanges_ispolnenie_3": "[ГОСТ 12815] Исполнение 3 (с впадиной)",
    "gost_12815_flanges_ispolneniya_upoltnitelnyh_poverkhnostey": "[ГОСТ 12815] Исполнения уплотнительных поверхностей",
    "gost_12815_flanges_material_flantsov": "[ГОСТ 12815] Материал фланцев",
    "gost_12815_flanges_kruglye_stalnye_flantsy": "[ГОСТ 12815] Круглые стальные фланцы",
    "gost_12815_flanges_korpus_diskovogo_zatvora_dn80": "[ГОСТ 12815] Корпус затвора DN80 (фланцевый стык)",
    "gost_12815_flanges_obratny_klapan": "[ГОСТ 12815] Клапан обратный (сопряжение фланцев)"
}

PRUNE_NODE_IDS = {
    "gost_12815_flanges_ru",
    "gost_12815_flanges_mpa",
    "gost_12815_flanges_kgs_sm2"
}

def _enrich_graph_json(graph_json_path: str):
    """Enriches node names with official standard numbers and prunes measurement units noise."""
    import json
    try:
        with open(graph_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data['nodes'] = [n for n in data.get('nodes', []) if n.get('id') not in PRUNE_NODE_IDS]
        data['links'] = [l for l in data.get('links', []) if l.get('source') not in PRUNE_NODE_IDS and l.get('target') not in PRUNE_NODE_IDS]

        for n in data['nodes']:
            nid = n.get('id', '')
            old_lbl = n.get('label', '')
            sf = str(n.get('source_file') or '')

            if nid in STANDARDS_MAP:
                n['label'] = STANDARDS_MAP[nid]
            else:
                if '12815' in nid or '12815' in sf:
                    if not old_lbl.startswith('[ГОСТ'):
                        n['label'] = f"[ГОСТ 12815] {old_lbl}"
                elif '6111' in nid or '6111' in sf:
                    if not old_lbl.startswith('[ГОСТ'):
                        n['label'] = f"[ГОСТ 6111] {old_lbl}"
                elif '28343' in nid or '28343' in sf:
                    if not old_lbl.startswith('[ГОСТ'):
                        n['label'] = f"[ГОСТ 28343] {old_lbl}"
                elif '9544' in nid or '9544' in sf:
                    if not old_lbl.startswith('[ГОСТ'):
                        n['label'] = f"[ГОСТ 9544] {old_lbl}"
                elif 'bom' in sf.lower():
                    if not old_lbl.startswith('[BOM'):
                        n['label'] = f"[BOM] {old_lbl}"
                elif 'dfmea' in sf.lower():
                    if not old_lbl.startswith('[DFMEA'):
                        n['label'] = f"[DFMEA] {old_lbl}"
                elif 'defect' in sf.lower():
                    if not old_lbl.startswith('[Акт'):
                        n['label'] = f"[Акт] {old_lbl}"

        for l in data['links']:
            r = l.get('relation', '')
            if r == 'references':
                l['relation_ru'] = 'регламентирует / ссылается на'
            elif r == 'conceptually_related_to':
                l['relation_ru'] = 'включает в себя'
            else:
                l['relation_ru'] = 'связан с'

        with open(graph_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[GraphSync] Enriched {len(data['nodes'])} nodes and {len(data['links'])} links in graph.json.")
    except Exception as e:
        logger.warning(f"[GraphSync] Could not enrich graph.json: {e}")


def _enrich_graph_html(graph_html_path: str):
    """Enriches exported HTML with sharp arrows, Russian edge labels, tooltips and directed sidebar."""
    import re
    try:
        with open(graph_html_path, "r", encoding="utf-8") as f:
            html_str = f.read()

        old_edges_pattern = r"const edgesDS = new vis\.DataSet\(RAW_EDGES\.map\(\(e, i\) => \(\{\s*id: i, from: e\.from, to: e\.to,\s*label: '',.*?\n\}\)\)\);"
        new_edges_code = """const edgesDS = new vis.DataSet(RAW_EDGES.map((e, i) => {
  let relRu = 'ссылается на';
  if (e.label === 'conceptually_related_to' || e.label === 'relates_to') relRu = 'включает в себя';
  else if (e.label === 'contains' || e.label === 'includes') relRu = 'включает деталь';
  else if (e.label === 'regulates') relRu = 'регламентирует';
  else if (e.label === 'causes') relRu = 'вызывает дефект';

  const srcNode = RAW_NODES.find(n => n.id === e.from);
  const tgtNode = RAW_NODES.find(n => n.id === e.to);
  const srcLbl = srcNode ? srcNode.label : e.from;
  const tgtLbl = tgtNode ? tgtNode.label : e.to;
  const fullTooltip = `➡️ НАПРАВЛЕНИЕ СВЯЗИ:\\n«${srcLbl}»\\n  └──[ ${relRu} ]──▶\\n«${tgtLbl}»`;

  return {
    id: i,
    from: e.from,
    to: e.to,
    label: relRu,
    title: fullTooltip,
    dashes: e.dashes,
    width: Math.max(e.width || 2, 2.2),
    color: e.color || { color: '#8888bb', highlight: '#ae81ff', opacity: 0.8 },
    font: { size: 10, color: '#c0caf5', strokeWidth: 3, strokeColor: '#1a1a2e', align: 'middle' },
    arrows: {
      to: { enabled: true, scaleFactor: 0.85, type: 'arrow' }
    }
  };
}));"""

        html_str = re.sub(old_edges_pattern, new_edges_code, html_str, flags=re.DOTALL)

        old_show_info = r"function showInfo\(nodeId\) \{.*?(?=\n\n|\nnetwork\.on)"
        new_show_info = """function showInfo(nodeId) {
  const n = nodesDS.get(nodeId);
  if (!n) return;

  const connectedEdges = RAW_EDGES.filter(e => e.from === nodeId || e.to === nodeId);
  
  const outgoing = [];
  const incoming = [];

  connectedEdges.forEach(e => {
    let relRu = 'ссылается на';
    if (e.label === 'conceptually_related_to' || e.label === 'relates_to') relRu = 'включает в себя';
    else if (e.label === 'contains' || e.label === 'includes') relRu = 'включает деталь';
    else if (e.label === 'regulates') relRu = 'регламентирует';

    if (e.from === nodeId) {
      const targetNode = nodesDS.get(e.to);
      if (targetNode) {
        outgoing.push({ node: targetNode, rel: relRu });
      }
    } else {
      const sourceNode = nodesDS.get(e.from);
      if (sourceNode) {
        incoming.push({ node: sourceNode, rel: relRu });
      }
    }
  });

  const esc = str => (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const renderNeighbor = item => `
    <div class="neighbor-link" style="border-left-color:${item.node.color ? item.node.color.background : '#ae81ff'}; margin-bottom:4px; padding:4px 6px;" data-nid="${esc(item.node.id)}">
      <span style="font-size:10px; color:#8f908a; display:block;">[${esc(item.rel)}]</span>
      <b>${esc(item.node.label)}</b>
    </div>`;

  let connectionsHtml = '';
  if (outgoing.length > 0) {
    connectionsHtml += `<div style="margin-top:10px; font-weight:700; color:#66d9ef; font-size:11px; text-transform:uppercase;">➡️ Исходящие (узел регламентирует / ссылается):</div>
      <div style="max-height:130px; overflow-y:auto; margin-top:4px;">${outgoing.map(renderNeighbor).join('')}</div>`;
  }
  if (incoming.length > 0) {
    connectionsHtml += `<div style="margin-top:10px; font-weight:700; color:#a6e22e; font-size:11px; text-transform:uppercase;">⬅️ Входящие (на этот узел ссылаются):</div>
      <div style="max-height:130px; overflow-y:auto; margin-top:4px;">${incoming.map(renderNeighbor).join('')}</div>`;
  }

  document.getElementById('info-content').innerHTML = `
    <div class="field" style="font-size:13px; color:#fd971f;"><b>${esc(n.label)}</b></div>
    <div class="field"><b>Тип:</b> ${esc(n._file_type || 'концепт')}</div>
    <div class="field"><b>Кластер:</b> ${esc(n._community_name)}</div>
    <div class="field"><b>Источник:</b> ${esc(n._source_file || '-')}</div>
    <div class="field"><b>Степень связности:</b> ${n._degree} связей</div>
    ${connectionsHtml}
  `;

  document.querySelectorAll('.neighbor-link').forEach(el => {
    el.addEventListener('click', () => {
      const nid = el.getAttribute('data-nid');
      network.selectNodes([nid]);
      showInfo(nid);
      network.focus(nid, { scale: 1.2, animation: true });
    });
  });
}"""

        html_str = re.sub(old_show_info, new_show_info, html_str, flags=re.DOTALL)

        with open(graph_html_path, "w", encoding="utf-8") as f:
            f.write(html_str)
        logger.info("[GraphSync] Successfully enriched graph.html with arrows, labels, and directed sidebar.")
    except Exception as e:
        logger.warning(f"[GraphSync] Could not enrich graph.html: {e}")

