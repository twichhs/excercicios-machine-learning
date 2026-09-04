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


# Um emoji por tema, so para navegacao visual do indice.
EMOJI_TEMA = {
    "01-estatistica": "📊",
    "02-algebra-linear-e-otimizacao": "🔢",
    "03-preparacao-de-dados": "🧹",
    "04-aprendizado-supervisionado": "🎯",
    "05-aprendizado-nao-supervisionado": "🔍",
    "06-avaliacao-e-validacao": "⚖️",
    "07-series-temporais": "📈",
    "08-deep-learning": "🧠",
    "09-nlp-e-llms": "💬",
    "10-inferencia-causal": "🔗",
    "11-interpretabilidade-e-fairness": "🔬",
    "12-sistemas-de-recomendacao": "🛒",
    "13-mlops-e-producao": "🚀",
}

CICLO = [
    "```",
    "   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐",
    "   │  📕  LEIA       │     │  💻  RODE       │     │  ✏️  RESOLVA    │",
    "   │                 │ ──▶ │                 │ ──▶ │                 │",
    "   │   teoria.pdf    │     │ notebooks-guia  │     │  99-exercicios  │",
    "   └─────────────────┘     └─────────────────┘     └─────────────────┘",
    "     entenda a ideia         mexa nos números        agora sem apoio",
    "```",
]


def _celula(texto: str) -> str:
    """Escapa o que quebraria uma celula de tabela markdown."""
    return texto.replace("|", "\\|").replace("\n", " ")


def _readme_tema(tema) -> None:
    emoji = EMOJI_TEMA.get(tema.slug, "📁")
    linhas = [
        f"# {emoji} {tema.titulo}", "",
        f"> {tema.resumo}", "",
        f"**{len(tema.modulos)} módulos** · leia de cima para baixo — cada um "
        "assume o anterior.", "",
        "---", "",
        "## 📚 Módulos deste tema", "",
        "| # | Módulo | O que você vai aprender |",
        "| :-: | :-- | :-- |",
    ]
    for k, m in enumerate(tema.modulos, start=1):
        linhas.append(f"| **{k}** | **[{m.titulo}]({m.slug}/teoria.pdf)** "
                      f"| {_celula(m.resumo)} |")

    linhas += ["", "---", "", "## 🗂️ Arquivos de cada módulo", "",
               "| Módulo | 📕 Teoria | 💻 Notebooks-guia | ✏️ Prática |",
               "| :-- | :-: | :-- | :-: |"]
    for m in tema.modulos:
        guias = [s for s, _ in m.notebooks if s != "99-exercicios"]
        nbs = " · ".join(f"[{k + 1}]({m.slug}/{s}.ipynb)" for k, s in enumerate(guias))
        linhas.append(
            f"| {m.titulo} | [PDF]({m.slug}/teoria.pdf) | {nbs} "
            f"| [abrir]({m.slug}/99-exercicios.ipynb) |")

    linhas += [
        "", "---", "", "## 🔄 Como estudar cada módulo", "", *CICLO, "",
        "> 💡 Os notebooks-guia **já vêm com as saídas prontas** — dá para ler "
        "sem rodar nada. Mas o material foi escrito para ser alterado: mude um "
        "parâmetro, rode de novo, veja o que quebra.", "",
        "> ✏️ O `99-exercicios` é o único lugar onde o código é seu. Cada "
        "exercício tem o gabarito logo abaixo — resolva **antes** de rolar a "
        "página, porque ler a solução dá a sensação de entender sem o "
        "entendimento.", "",
        "---", "", "[⬅️ Voltar para o índice geral](../README.md)", "",
    ]
    (RAIZ / tema.slug / "README.md").write_text("\n".join(linhas), encoding="utf-8")


def _readme_raiz() -> None:
    n_mod = sum(len(t.modulos) for t in CURRICULO)
    n_nb = sum(len(m.notebooks) for _, m in todos_os_modulos())
    linhas = [
        "# 📚 Machine Learning & Deep Learning", "",
        "> Uma trilha de estudos completa para virar cientista de dados — da "
        "primeira média aritmética até um modelo rodando em produção.", "",
        f"### 🧭 {len(CURRICULO)} temas · {n_mod} módulos · "
        f"{n_nb - n_mod} notebooks-guia · {n_mod} notebooks de exercícios", "",
        "Escrito para quem tem **Python intermediário** e **estatística "
        "superficial**. Nenhum conceito estatístico aparece sem ser construído "
        "do zero antes — se você não sabe o que é um desvio-padrão, comece pelo "
        "tema 1 e siga a ordem.", "",
        "---", "",
        "## 🚀 Comece por aqui", "",
        "### 1️⃣ Prepare o ambiente (uma vez só)", "",
        "```bash",
        "python3 -m venv .venv",
        "source .venv/bin/activate",
        "pip install -r requirements.txt",
        "python -m ipykernel install --user --name curso-ml \\",
        '    --display-name "Python (curso ML)"',
        "jupyter lab",
        "```", "",
        "### 2️⃣ Abra o primeiro módulo", "",
        "```",
        "01-estatistica/01-fundamentos-e-estatistica-descritiva/",
        "```", "",
        "### 3️⃣ Siga sempre o mesmo ciclo", "", *CICLO, "",
        "---", "",
        "## 🗂️ O que tem dentro de cada módulo", "",
        "| Arquivo | O que é | Como usar |",
        "| :-- | :-- | :-- |",
        "| 📕 `teoria.pdf` | O material denso: conceitos, fórmulas, analogias e "
        "aplicações reais de mercado. | Leia um capítulo por vez, sem pressa. |",
        "| 💻 `01-*.ipynb`, `02-*.ipynb`… | Notebooks-guia, muito comentados e já "
        "com as saídas embutidas. | Rode, mude os parâmetros, veja o que muda. |",
        "| ✏️ `99-exercicios.ipynb` | Exercícios do módulo, com gabarito "
        "comentado logo abaixo de cada um. | Resolva **antes** de olhar a "
        "resposta. |",
        "| 📝 `teoria.md` | A fonte de onde o PDF é gerado. | Só interessa se "
        "você for editar o material. |",
        "", "",
        "> 💡 **Como saber se entendeu?** Se você consegue resolver o "
        "`99-exercicios` sem olhar o gabarito, entendeu. Se não consegue, volte "
        "ao PDF — não adianta seguir em frente, porque o próximo módulo assume "
        "este.", "",
        "> 🟢 🟡 🔴 Os exercícios são marcados por dificuldade: **base**, "
        "**aplicação** e **síntese**. Se o tempo estiver curto, faça os 🟢 e 🟡 "
        "de todos os módulos antes de voltar aos 🔴.", "",
        "---", "",
        "## 🗺️ A trilha completa", "",
    ]
    for i, tema in enumerate(CURRICULO, start=1):
        emoji = EMOJI_TEMA.get(tema.slug, "📁")
        linhas += [
            f"### {emoji} {i}. {tema.titulo}", "",
            f"> {tema.resumo}", "",
            "| Módulo | O que você vai aprender |",
            "| :-- | :-- |",
        ]
        for m in tema.modulos:
            linhas.append(f"| **[{m.titulo}]({tema.slug}/{m.slug}/teoria.pdf)** "
                          f"| {_celula(m.resumo)} |")
        linhas += ["", f"📂 **[Abrir o tema completo]({tema.slug}/README.md)**", "",
                   "---", ""]

    linhas += [
        "## 🛠️ Reconstruindo o material", "",
        "Só é necessário se você for **editar** o conteúdo. Para estudar, basta "
        "abrir os arquivos.", "",
        "```bash",
        "python _ferramentas/build.py status      # o que já existe e o que falta",
        "python _ferramentas/build.py tudo        # estrutura + PDFs + notebooks",
        "python _ferramentas/build.py pdfs 01-estatistica   # só um tema",
        "```", "",
        "### ⚙️ Como o material é construído", "",
        "| Etapa | Ferramenta | Detalhe |",
        "| :-- | :-- | :-- |",
        "| Estrutura e índices | `_ferramentas/curriculo.py` | Fonte única de "
        "verdade: temas, módulos e notebooks. |",
        "| PDFs | `_ferramentas/md2pdf.py` | ReportLab para o layout e o motor "
        "`mathtext` do Matplotlib para as fórmulas — **sem depender de LaTeX**. |",
        "| Notebooks | `_ferramentas/py2nb.py` | Os notebooks são escritos como "
        "scripts em `_fontes/` e só viram `.ipynb` **depois de executarem sem "
        "erro**. Nenhum notebook chega quebrado. |",
        "", "---", "",
        "## ❓ Dúvidas frequentes", "",
        "**Preciso saber matemática avançada?**  ",
        "Não. Álgebra do ensino médio basta. O tema 2 constrói a álgebra linear "
        "necessária a partir da geometria.", "",
        "**Posso pular temas?**  ",
        "Os temas 1 a 3 são pré-requisito de tudo. A partir do tema 4, dá para "
        "escolher — mas Deep Learning (8) assume Otimização (2), e NLP (9) "
        "assume Deep Learning.", "",
        "**Os notebooks precisam ser executados?**  ",
        "Não para ler: as saídas já vêm embutidas. Sim para aprender: o material "
        "foi feito para ser alterado.", "",
        "**Quanto tempo leva?**  ",
        "Cada módulo tem entre 8 e 12 horas de leitura mais prática. São 54 "
        "módulos — trate como uma maratona de meses, não um fim de semana.", "",
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
