"""
build.py — Orquestrador do material do curso.

    python _ferramentas/build.py estrutura   # cria pastas, READMEs e indices
    python _ferramentas/build.py pdfs        # compila todos os teoria.md
    python _ferramentas/build.py notebooks   # gera e executa todos os .ipynb
    python _ferramentas/build.py tudo        # os tres, na ordem
    python _ferramentas/build.py status      # o que ja existe e o que falta

Filtro opcional por prefixo do tema/modulo:
    python _ferramentas/build.py pdfs 01-estatistica
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from curriculo import CURRICULO, todos_os_modulos  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
FONTES = RAIZ / "_fontes"
PY = sys.executable


# --------------------------------------------------------------------------
def cria_estrutura(filtro: str = "") -> None:
    for tema in CURRICULO:
        if filtro and not tema.slug.startswith(filtro):
            continue
        (RAIZ / tema.slug).mkdir(exist_ok=True)
        for modulo in tema.modulos:
            (RAIZ / tema.slug / modulo.slug).mkdir(exist_ok=True)
            (FONTES / tema.slug / modulo.slug).mkdir(parents=True, exist_ok=True)
        _readme_tema(tema)
    _readme_raiz()
    print(f"estrutura pronta: {len(CURRICULO)} temas, "
          f"{sum(len(t.modulos) for t in CURRICULO)} módulos")


def _readme_tema(tema) -> None:
    linhas = [
        f"# {tema.titulo}", "",
        tema.resumo, "",
        "| Módulo | Conteúdo | Teoria | Notebooks-guia | Prática |",
        "|---|---|---|---|---|",
    ]
    for m in tema.modulos:
        pdf = f"[PDF]({m.slug}/teoria.pdf)"
        guias = [s for s, _ in m.notebooks if s != "99-exercicios"]
        nbs = ", ".join(f"[{k+1}]({m.slug}/{s}.ipynb)" for k, s in enumerate(guias))
        ex = f"[exercícios]({m.slug}/99-exercicios.ipynb)"
        linhas.append(f"| **{m.titulo}** | {m.resumo} | {pdf} | {nbs} | {ex} |")
    linhas += [
        "", "---", "",
        "Cada módulo tem um `teoria.pdf` (denso, com fórmulas e aplicações de "
        "mercado), notebooks-guia executáveis e um notebook de **exercícios**. "
        "Sugestão de uso: leia o PDF até o fim de um capítulo, rode o notebook-guia "
        "correspondente mexendo nos parâmetros, e só então abra os exercícios — "
        "eles são o único lugar onde o código é seu.", "",
        "[← voltar ao índice geral](../README.md)", "",
    ]
    (RAIZ / tema.slug / "README.md").write_text("\n".join(linhas), encoding="utf-8")


def _readme_raiz() -> None:
    n_mod = sum(len(t.modulos) for t in CURRICULO)
    n_nb = sum(len(m.notebooks) for _, m in todos_os_modulos())
    linhas = [
        "# Machine Learning & Deep Learning — Material do Curso", "",
        "Material completo para formação de cientistas de dados, escrito para "
        "quem tem **Python intermediário** e **estatística superficial**. Cada "
        "conceito estatístico é construído do zero antes de ser usado.", "",
        f"**{len(CURRICULO)} temas · {n_mod} módulos · {n_nb - n_mod} notebooks-guia "
        f"· {n_mod} notebooks de exercícios**", "",
        "## Como o material está organizado", "",
        "```",
        "<tema>/",
        "  <módulo>/",
        "    teoria.md      ← fonte do material teórico",
        "    teoria.pdf     ← PDF denso: conceitos, fórmulas, aplicações reais",
        "    NN-*.ipynb     ← notebooks-guia executáveis, muito comentados",
        "    99-exercicios.ipynb  ← exercícios do módulo, com gabarito comentado",
        "```", "",
        "## Índice", "",
    ]
    for tema in CURRICULO:
        linhas.append(f"### [{tema.titulo}]({tema.slug}/README.md)")
        linhas.append("")
        linhas.append(f"_{tema.resumo}_")
        linhas.append("")
        for m in tema.modulos:
            linhas.append(
                f"- **[{m.titulo}]({tema.slug}/{m.slug}/teoria.pdf)** — {m.resumo}")
        linhas.append("")
    linhas += [
        "## Preparando o ambiente", "",
        "```bash",
        "python3 -m venv .venv",
        "source .venv/bin/activate",
        "pip install -r requirements.txt",
        "python -m ipykernel install --user --name curso-ml \\",
        '    --display-name "Python (curso ML)"',
        "jupyter lab",
        "```", "",
        "## Reconstruindo o material", "",
        "```bash",
        "python _ferramentas/build.py tudo        # estrutura + PDFs + notebooks",
        "python _ferramentas/build.py pdfs 01-estatistica   # só um tema",
        "python _ferramentas/build.py status      # o que falta",
        "```", "",
        "Os PDFs são gerados por uma toolchain própria (`_ferramentas/md2pdf.py`) "
        "que usa ReportLab para o layout e o motor `mathtext` do Matplotlib para "
        "renderizar as fórmulas — sem depender de LaTeX instalado.", "",
        "Os notebooks são escritos como scripts em `_fontes/` (formato *percent*) "
        "e só viram `.ipynb` **depois de executarem sem erro**, com as saídas "
        "embutidas. Nenhum notebook do curso chega ao aluno quebrado.", "",
    ]
    (RAIZ / "README.md").write_text("\n".join(linhas), encoding="utf-8")


# --------------------------------------------------------------------------
def compila_pdfs(filtro: str = "") -> int:
    alvos = []
    for tema, modulo in todos_os_modulos():
        if filtro and not f"{tema.slug}/{modulo.slug}".startswith(filtro):
            continue
        md = RAIZ / tema.slug / modulo.slug / "teoria.md"
        if md.exists():
            alvos.append(md)
    if not alvos:
        print("nenhum teoria.md encontrado para esse filtro")
        return 0
    print(f"compilando {len(alvos)} PDFs…")
    return subprocess.call([PY, str(RAIZ / "_ferramentas" / "md2pdf.py"), *map(str, alvos)])


def gera_notebooks(filtro: str = "") -> int:
    falhas = 0
    for tema, modulo in todos_os_modulos():
        if filtro and not f"{tema.slug}/{modulo.slug}".startswith(filtro):
            continue
        for slug, _titulo in modulo.notebooks:
            fonte = FONTES / tema.slug / modulo.slug / f"{slug}.py"
            if not fonte.exists():
                continue
            destino = RAIZ / tema.slug / modulo.slug / f"{slug}.ipynb"
            print(f"  → {tema.slug}/{modulo.slug}/{slug}")
            cod = subprocess.call(
                [PY, str(RAIZ / "_ferramentas" / "py2nb.py"), str(fonte), str(destino)])
            if cod != 0:
                falhas += 1
    if falhas:
        print(f"\n{falhas} notebook(s) falharam", file=sys.stderr)
    return 1 if falhas else 0


def status(filtro: str = "") -> None:
    print(f"{'módulo':58s} {'teoria':>7s} {'pdf':>5s} {'notebooks':>12s}")
    print("-" * 86)
    tot_md = tot_pdf = tot_nb = tot_nb_esp = 0
    for tema, modulo in todos_os_modulos():
        cam = f"{tema.slug}/{modulo.slug}"
        if filtro and not cam.startswith(filtro):
            continue
        d = RAIZ / cam
        tem_md = (d / "teoria.md").exists()
        tem_pdf = (d / "teoria.pdf").exists()
        feitos = sum(1 for s, _ in modulo.notebooks if (d / f"{s}.ipynb").exists())
        esperados = len(modulo.notebooks)
        tot_md += tem_md
        tot_pdf += tem_pdf
        tot_nb += feitos
        tot_nb_esp += esperados
        print(f"{cam:58s} {'sim' if tem_md else '—':>7s} "
              f"{'sim' if tem_pdf else '—':>5s} {feitos:>7d}/{esperados:<4d}")
    print("-" * 86)
    n = sum(1 for t, m in todos_os_modulos()
            if not filtro or f"{t.slug}/{m.slug}".startswith(filtro))
    print(f"{'TOTAL':58s} {tot_md:>4d}/{n:<2d} {tot_pdf:>2d}/{n:<2d} "
          f"{tot_nb:>7d}/{tot_nb_esp:<4d}")


# --------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    comando = argv[1] if len(argv) > 1 else "status"
    filtro = argv[2] if len(argv) > 2 else ""
    if comando == "estrutura":
        cria_estrutura(filtro)
        return 0
    if comando == "pdfs":
        return compila_pdfs(filtro)
    if comando == "notebooks":
        return gera_notebooks(filtro)
    if comando == "tudo":
        cria_estrutura(filtro)
        return compila_pdfs(filtro) or gera_notebooks(filtro)
    if comando == "status":
        status(filtro)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
