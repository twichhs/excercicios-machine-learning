"""
md2pdf.py — Conversor Markdown -> PDF para o material teorico do curso.

Por que uma ferramenta propria? O ambiente nao tem LaTeX nem pandoc. Entao:
  - o layout do PDF e montado com ReportLab (Platypus);
  - as formulas matematicas sao renderizadas pelo motor `mathtext` do
    Matplotlib e embutidas como imagens PNG de alta resolucao (300 dpi).

Sintaxe Markdown suportada (subconjunto deliberadamente pequeno e previsivel):

  # Titulo do documento          -> capa
  ## Capitulo                    -> numerado 1., 2., ... (quebra de pagina)
  ### Secao                      -> numerada 1.1, 1.2, ...
  #### Subsecao                  -> numerada 1.1.1, ...

  **negrito**  *italico*  `codigo`  $formula inline$
  $$formula em display$$
  ```python ... ```             -> bloco de codigo com realce
  - item / 1. item              -> listas (2 niveis)
  | a | b |                     -> tabelas com pipe
  ---                           -> linha horizontal
  > [!NOTA] / [!MERCADO] / [!ARMADILHA] / [!ANALOGIA] / [!FORMULA]
                                -> caixas de destaque coloridas

IMPORTANTE sobre as formulas: o `mathtext` aceita um subconjunto amplo do
LaTeX -- \\frac, \\sum, \\prod, \\int, \\sqrt, \\binom, \\text, \\operatorname,
\\mathbb, \\mathcal, \\mathbf, \\boldsymbol, \\underset, \\overset,
\\left(...\\right), indices e expoentes. NAO funcionam AMBIENTES: nada de
\\begin{align} ou \\begin{pmatrix} -- matrizes precisam ser escritas em bloco de
codigo ou como imagem. Apelidos comuns (\\le, \\ge, \\iff, \\lVert, \\stackrel)
sao traduzidos automaticamente por `normaliza_latex`.
O build falha alto e claro se uma formula nao compilar.
"""
from __future__ import annotations

import hashlib
import html
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, CondPageBreak, Frame, HRFlowable, Image as RLImage,
    KeepTogether, ListFlowable, ListItem, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

RAIZ = Path(__file__).resolve().parent.parent
CACHE = RAIZ / "_assets" / "cache_formulas"
CACHE.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Paleta -- tons sobrios, alto contraste na impressao em preto e branco
# --------------------------------------------------------------------------
AZUL_ESCURO = colors.HexColor("#12314F")
AZUL = colors.HexColor("#1F5C8B")
AZUL_CLARO = colors.HexColor("#EAF1F7")
CINZA_TEXTO = colors.HexColor("#1C1C1C")
CINZA_MEDIO = colors.HexColor("#5A5A5A")
CINZA_LINHA = colors.HexColor("#D5D9DE")
CODIGO_BG = colors.HexColor("#F5F6F8")
VERDE = colors.HexColor("#1E6B4F")
VERDE_BG = colors.HexColor("#E9F4EF")
VERMELHO = colors.HexColor("#9C2B2B")
VERMELHO_BG = colors.HexColor("#FBECEC")
ROXO = colors.HexColor("#5B3E86")
ROXO_BG = colors.HexColor("#F1ECF9")
AMBAR = colors.HexColor("#8A6100")
AMBAR_BG = colors.HexColor("#FDF4E3")

# --------------------------------------------------------------------------
# Fontes: DejaVu vem junto com o Matplotlib e cobre acentuacao + simbolos
# --------------------------------------------------------------------------
_MPL_FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def _registrar_fontes() -> None:
    mapa = {
        "DejaVu": "DejaVuSans.ttf",
        "DejaVu-Bold": "DejaVuSans-Bold.ttf",
        "DejaVu-Italic": "DejaVuSans-Oblique.ttf",
        "DejaVu-BoldItalic": "DejaVuSans-BoldOblique.ttf",
        "DejaVuMono": "DejaVuSansMono.ttf",
        "DejaVuMono-Bold": "DejaVuSansMono-Bold.ttf",
        "DejaVuMono-Italic": "DejaVuSansMono-Oblique.ttf",
    }
    for nome, arquivo in mapa.items():
        pdfmetrics.registerFont(TTFont(nome, str(_MPL_FONTS / arquivo)))
    pdfmetrics.registerFontFamily(
        "DejaVu", normal="DejaVu", bold="DejaVu-Bold",
        italic="DejaVu-Italic", boldItalic="DejaVu-BoldItalic")
    pdfmetrics.registerFontFamily(
        "DejaVuMono", normal="DejaVuMono", bold="DejaVuMono-Bold",
        italic="DejaVuMono-Italic", boldItalic="DejaVuMono-Bold")


_registrar_fontes()

# --------------------------------------------------------------------------
# Renderizacao de formulas (mathtext -> PNG, com cache em disco)
# --------------------------------------------------------------------------
_DPI = 320
_parser_math = mathtext.MathTextParser("path")


class ErroDeFormula(RuntimeError):
    pass


# O mathtext aceita quase todo o LaTeX matematico do dia a dia, mas rejeita
# alguns apelidos comuns. Normalizamos aqui para que o material possa ser
# escrito com a notacao usual, sem o autor precisar decorar as excecoes.
_ALIAS = [
    (re.compile(r"\\le\b(?!q)"), r"\\leq"),
    (re.compile(r"\\ge\b(?!q)"), r"\\geq"),
    (re.compile(r"\\iff\b"), r"\\Leftrightarrow"),
    (re.compile(r"\\implies\b"), r"\\Rightarrow"),
    (re.compile(r"\\lVert\b"), r"\\|"),
    (re.compile(r"\\rVert\b"), r"\\|"),
    (re.compile(r"\\stackrel\b"), r"\\overset"),
    (re.compile(r"\\coloneqq\b"), r":="),
]


def normaliza_latex(latex: str) -> str:
    for rx, sub in _ALIAS:
        latex = rx.sub(sub, latex)
    return latex


def renderiza_formula(latex: str, tamanho_pt: float) -> tuple[str, float, float, float]:
    """Renderiza `latex` e devolve (caminho_png, largura_pt, altura_pt, base_pt).

    `base_pt` e a profundidade: quantos pontos a imagem desce ABAIXO da linha
    de base do texto. O `math_to_image` monta a figura com exatamente a caixa
    da formula (sem padding) e a base a `depth` pontos do rodape da imagem --
    e por isso conseguimos alinhamento tipografico exato, e nao um chute.

    O cache e indexado pelo hash do par (formula, tamanho); rebuilds do
    material ficam praticamente instantaneos.
    """
    latex = normaliza_latex(latex)
    chave = hashlib.sha1(f"{latex}||{tamanho_pt}".encode()).hexdigest()[:20]
    png = CACHE / f"{chave}.png"
    meta = CACHE / f"{chave}.txt"
    if png.exists() and meta.exists():
        profundidade = float(meta.read_text())
    else:
        try:
            profundidade = float(mathtext.math_to_image(
                f"${latex}$", str(png),
                prop=FontProperties(size=tamanho_pt), dpi=_DPI, format="png"))
        except Exception as exc:  # noqa: BLE001
            raise ErroDeFormula(
                f"mathtext nao conseguiu compilar a formula:\n  {latex}\n  -> {exc}"
            ) from exc
        meta.write_text(str(profundidade))
    with Image.open(png) as img:
        larg_px, alt_px = img.size
    escala = 72.0 / _DPI
    return str(png), larg_px * escala, alt_px * escala, profundidade


# --------------------------------------------------------------------------
# Estilos de paragrafo
# --------------------------------------------------------------------------
_base = getSampleStyleSheet()

E = {
    "corpo": ParagraphStyle(
        "corpo", parent=_base["BodyText"], fontName="DejaVu", fontSize=10,
        leading=15.5, alignment=TA_JUSTIFY, textColor=CINZA_TEXTO,
        spaceBefore=0, spaceAfter=7),
    "h1": ParagraphStyle(
        "h1", fontName="DejaVu-Bold", fontSize=19, leading=24,
        textColor=AZUL_ESCURO, spaceBefore=0, spaceAfter=12),
    "h2": ParagraphStyle(
        "h2", fontName="DejaVu-Bold", fontSize=13.5, leading=18,
        textColor=AZUL, spaceBefore=16, spaceAfter=7),
    "h3": ParagraphStyle(
        "h3", fontName="DejaVu-Bold", fontSize=11, leading=15,
        textColor=CINZA_TEXTO, spaceBefore=11, spaceAfter=5),
    "item": ParagraphStyle(
        "item", fontName="DejaVu", fontSize=10, leading=15,
        textColor=CINZA_TEXTO, alignment=TA_LEFT, spaceAfter=3),
    "codigo": ParagraphStyle(
        "codigo", fontName="DejaVuMono", fontSize=8.1, leading=11.4,
        textColor=colors.HexColor("#20262E"), spaceBefore=0, spaceAfter=0),
    "legenda": ParagraphStyle(
        "legenda", fontName="DejaVu-Italic", fontSize=8.6, leading=12,
        textColor=CINZA_MEDIO, alignment=TA_CENTER, spaceBefore=3, spaceAfter=8),
    "tabela": ParagraphStyle(
        "tabela", fontName="DejaVu", fontSize=8.6, leading=12,
        textColor=CINZA_TEXTO),
    "tabela_cab": ParagraphStyle(
        "tabela_cab", fontName="DejaVu-Bold", fontSize=8.6, leading=12,
        textColor=colors.white),
    "caixa": ParagraphStyle(
        "caixa", fontName="DejaVu", fontSize=9.4, leading=14,
        textColor=CINZA_TEXTO, alignment=TA_JUSTIFY, spaceAfter=4),
    "caixa_titulo": ParagraphStyle(
        "caixa_titulo", fontName="DejaVu-Bold", fontSize=8.4, leading=12,
        spaceAfter=3),
    "capa_titulo": ParagraphStyle(
        "capa_titulo", fontName="DejaVu-Bold", fontSize=27, leading=33,
        textColor=AZUL_ESCURO, alignment=TA_LEFT, spaceAfter=10),
    "capa_sub": ParagraphStyle(
        "capa_sub", fontName="DejaVu", fontSize=13, leading=19,
        textColor=AZUL, alignment=TA_LEFT, spaceAfter=6),
    "capa_meta": ParagraphStyle(
        "capa_meta", fontName="DejaVu", fontSize=9.5, leading=15,
        textColor=CINZA_MEDIO, alignment=TA_LEFT),
    "toc1": ParagraphStyle(
        "toc1", fontName="DejaVu-Bold", fontSize=10.5, leading=17,
        textColor=AZUL_ESCURO, spaceBefore=6),
    "toc2": ParagraphStyle(
        "toc2", fontName="DejaVu", fontSize=9.5, leading=14,
        textColor=CINZA_TEXTO, leftIndent=16),
}

CAIXAS = {
    "NOTA":      ("Nota",                 AZUL,     AZUL_CLARO),
    "MERCADO":   ("No mercado",           VERDE,    VERDE_BG),
    "ARMADILHA": ("Armadilha comum",      VERMELHO, VERMELHO_BG),
    "ANALOGIA":  ("Analogia",             ROXO,     ROXO_BG),
    "FORMULA":   ("Formulario",           AMBAR,    AMBAR_BG),
    "DEFINICAO": ("Definicao",            AZUL,     AZUL_CLARO),
}

# --------------------------------------------------------------------------
# Realce de sintaxe Python (regex simples, suficiente para leitura no papel)
# --------------------------------------------------------------------------
_KW = r"""\b(?:False|None|True|and|as|assert|async|await|break|class|continue|
def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|
nonlocal|not|or|pass|raise|return|try|while|with|yield|self)\b"""
_RE_KW = re.compile(_KW, re.X)
_RE_STR = re.compile(r"(\"\"\".*?\"\"\"|'''.*?'''|\"[^\"\n]*\"|'[^'\n]*')", re.S)
_RE_COM = re.compile(r"#[^\n]*")
_RE_NUM = re.compile(r"\b\d+\.?\d*(?:e-?\d+)?\b")
_RE_FUN = re.compile(r"\b([a-zA-Z_][a-zA-Z_0-9]*)\s*(?=\()")


def _realce(codigo: str, linguagem: str) -> str:
    """Devolve o codigo em markup ReportLab com cores por token."""
    marcas: list[tuple[int, int, str]] = []
    if linguagem in ("python", "py", ""):
        for rx, cor in ((_RE_STR, "#0A7B4E"), (_RE_COM, "#8A8F98")):
            for m in rx.finditer(codigo):
                marcas.append((m.start(), m.end(), cor))
        ocupado = lambda a, b: any(x < b and a < y for x, y, _ in marcas)  # noqa: E731
        for rx, cor in ((_RE_KW, "#A626A4"), (_RE_NUM, "#B36B00"),
                        (_RE_FUN, "#1F5C8B")):
            for m in rx.finditer(codigo):
                a, b = (m.span(1) if rx is _RE_FUN else m.span())
                if not ocupado(a, b):
                    marcas.append((a, b, cor))
    marcas.sort()

    def esc(t: str) -> str:
        # &nbsp; so DEPOIS do escape, e apenas no texto -- nunca nas tags,
        # senao o parser XML do ReportLab nao reconhece os atributos.
        return html.escape(t).replace(" ", "&nbsp;")

    saida, pos = [], 0
    for a, b, cor in marcas:
        if a < pos:
            continue
        saida.append(esc(codigo[pos:a]))
        saida.append(f'<font color="{cor}">{esc(codigo[a:b])}</font>')
        pos = b
    saida.append(esc(codigo[pos:]))
    return "".join(saida)


# --------------------------------------------------------------------------
# Formatacao inline: **negrito**, *italico*, `codigo`, $matematica$
# --------------------------------------------------------------------------
_RE_INLINE = re.compile(
    r"(?P<mat>(?<!\\)\$(?!\$)(?P<mat_c>(?:\\.|[^$\\])+?)\$)"
    r"|(?P<cod>`(?P<cod_c>[^`]+)`)"
    r"|(?P<neg>\*\*(?P<neg_c>[^*]+?)\*\*)"
    r"|(?P<ita>(?<!\*)\*(?P<ita_c>[^*\n]+?)\*(?!\*))"
)


def formata_inline(texto: str, tamanho_pt: float = 10.0) -> str:
    """Converte marcacao inline do Markdown para markup do ReportLab."""
    partes, pos = [], 0
    for m in _RE_INLINE.finditer(texto):
        partes.append(html.escape(texto[pos:m.start()]))
        if m.group("mat"):
            png, larg, alt, base = renderiza_formula(m.group("mat_c"), tamanho_pt)
            partes.append(
                f'<img src="{png}" width="{larg:.2f}" height="{alt:.2f}" '
                f'valign="{-base:.2f}"/>')
        elif m.group("cod"):
            partes.append(
                f'<font face="DejaVuMono" size="{tamanho_pt - 1.2:.1f}" '
                f'color="#A03A6B">{html.escape(m.group("cod_c"))}</font>')
        elif m.group("neg"):
            # recursivo: permite $formula$ e `codigo` dentro de **negrito**
            partes.append(f'<b>{formata_inline(m.group("neg_c"), tamanho_pt)}</b>')
        else:
            partes.append(f'<i>{formata_inline(m.group("ita_c"), tamanho_pt)}</i>')
        pos = m.end()
    partes.append(html.escape(texto[pos:]))
    return "".join(partes)


# --------------------------------------------------------------------------
# Blocos com fundo colorido (codigo, formula em display, caixas de destaque)
# --------------------------------------------------------------------------
def _bloco_colorido(conteudo, fundo, borda_esq=None, pad=7):
    """Empacota flowables numa tabela de 1 celula para ganhar fundo e borda."""
    estilo = [
        ("BACKGROUND", (0, 0), (-1, -1), fundo),
        ("LEFTPADDING", (0, 0), (-1, -1), pad + (4 if borda_esq else 0)),
        ("RIGHTPADDING", (0, 0), (-1, -1), pad),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    if borda_esq:
        estilo.append(("LINEBEFORE", (0, 0), (0, -1), 2.6, borda_esq))
    t = Table([[conteudo]], colWidths=[LARGURA_UTIL])
    t.setStyle(TableStyle(estilo))
    return t


LARGURA_UTIL = A4[0] - 2 * 2.3 * cm


class ImagemFormula(RLImage):
    """Formula em display, centralizada e reduzida se estourar a margem."""

    def __init__(self, latex: str, tamanho_pt: float = 14.0):
        png, larg, alt, _ = renderiza_formula(latex, tamanho_pt)
        maxw = LARGURA_UTIL - 30
        if larg > maxw:
            alt *= maxw / larg
            larg = maxw
        super().__init__(png, width=larg, height=alt)
        self.hAlign = "CENTER"


# --------------------------------------------------------------------------
# Parser de blocos
# --------------------------------------------------------------------------
class Documento:
    def __init__(self, texto_md: str, origem: str = "<md>"):
        self.linhas = texto_md.replace("\r\n", "\n").split("\n")
        self.origem = origem
        self.titulo = ""
        self.subtitulo = ""
        self.meta: dict[str, str] = {}
        self.flow: list = []
        self.n_cap = 0
        self.n_sec = 0
        self.n_sub = 0
        self._parse()

    # -- utilitarios ------------------------------------------------------
    def _p(self, texto: str, estilo="corpo"):
        return Paragraph(formata_inline(texto, E[estilo].fontSize), E[estilo])

    def _titulo_marcado(self, texto: str, estilo: str, nivel: int, chave: str):
        """Paragrafo de titulo que se registra no sumario."""
        par = Paragraph(formata_inline(texto, E[estilo].fontSize), E[estilo])
        par._toc_info = (nivel, texto, chave)  # lido pelo doc template
        return par

    # -- laco principal ---------------------------------------------------
    def _parse(self) -> None:
        i, n = 0, len(self.linhas)
        while i < n:
            linha = self.linhas[i]
            crua = linha.rstrip()

            # metadados no topo:  <!-- chave: valor -->
            if crua.startswith("<!--") and ":" in crua:
                corpo = crua.strip("<!->").strip()
                chave, _, valor = corpo.partition(":")
                self.meta[chave.strip().lower()] = valor.strip()
                i += 1
                continue

            if not crua.strip():
                i += 1
                continue

            # bloco de codigo
            if crua.startswith("```"):
                lang = crua[3:].strip().lower()
                i += 1
                buf = []
                while i < n and not self.linhas[i].startswith("```"):
                    buf.append(self.linhas[i])
                    i += 1
                i += 1
                self._add_codigo("\n".join(buf), lang)
                continue

            # formula em display
            if crua.strip() == "$$":
                i += 1
                buf = []
                while i < n and self.linhas[i].strip() != "$$":
                    buf.append(self.linhas[i])
                    i += 1
                i += 1
                self._add_formula(" ".join(x.strip() for x in buf))
                continue
            if crua.strip().startswith("$$") and crua.strip().endswith("$$") \
                    and len(crua.strip()) > 4:
                self._add_formula(crua.strip()[2:-2].strip())
                i += 1
                continue
            # "$$formula..." aberta numa linha e fechada em outra
            if crua.strip().startswith("$$"):
                buf = [crua.strip()[2:]]
                i += 1
                while i < n and not self.linhas[i].strip().endswith("$$"):
                    buf.append(self.linhas[i])
                    i += 1
                if i < n:
                    buf.append(self.linhas[i].strip()[:-2])
                    i += 1
                self._add_formula(" ".join(x.strip() for x in buf).strip())
                continue

            # titulos
            m = re.match(r"^(#{1,4})\s+(.*)$", crua)
            if m:
                self._add_titulo(len(m.group(1)), m.group(2).strip())
                i += 1
                continue

            # regua
            if re.fullmatch(r"-{3,}|\*{3,}", crua.strip()):
                self.flow.append(Spacer(1, 4))
                self.flow.append(HRFlowable(
                    width="100%", thickness=0.6, color=CINZA_LINHA,
                    spaceBefore=2, spaceAfter=8))
                i += 1
                continue

            # tabela
            if crua.lstrip().startswith("|") and i + 1 < n and \
                    re.match(r"^\s*\|[\s:|-]+\|\s*$", self.linhas[i + 1]):
                buf = []
                while i < n and self.linhas[i].lstrip().startswith("|"):
                    buf.append(self.linhas[i].strip())
                    i += 1
                self._add_tabela(buf)
                continue

            # caixa de destaque / citacao
            if crua.lstrip().startswith(">"):
                buf = []
                while i < n and self.linhas[i].lstrip().startswith(">"):
                    buf.append(re.sub(r"^\s*>\s?", "", self.linhas[i]))
                    i += 1
                self._add_caixa(buf)
                continue

            # listas
            if re.match(r"^\s*(?:[-*+]|\d+\.)\s+", crua):
                buf = []
                while i < n and (
                        re.match(r"^\s*(?:[-*+]|\d+\.)\s+", self.linhas[i])
                        or (self.linhas[i].startswith("   ")
                            and self.linhas[i].strip())):
                    buf.append(self.linhas[i])
                    i += 1
                self._add_lista(buf)
                continue

            # paragrafo: junta linhas ate encontrar linha em branco/novo bloco
            buf = []
            while i < n and self.linhas[i].strip() and not re.match(
                    r"^\s*(#{1,4}\s|```|\||>|-{3,}|\$\$|[-*+]\s|\d+\.\s)",
                    self.linhas[i]):
                buf.append(self.linhas[i].strip())
                i += 1
            if buf:
                self.flow.append(self._p(" ".join(buf)))

    # -- construtores de bloco -------------------------------------------
    def _add_titulo(self, nivel: int, texto: str) -> None:
        if nivel == 1:
            if not self.titulo:
                self.titulo = texto
            else:
                self.flow.append(PageBreak())
                self.flow.append(self._titulo_marcado(texto, "h1", 0, texto))
            return
        if nivel == 2:
            self.n_cap += 1
            self.n_sec = self.n_sub = 0
            rotulo = f"{self.n_cap}. {texto}"
            if self.flow:
                self.flow.append(PageBreak())
            self.flow.append(self._titulo_marcado(rotulo, "h1", 0, rotulo))
            self.flow.append(HRFlowable(
                width="100%", thickness=1.4, color=AZUL,
                spaceBefore=0, spaceAfter=12))
        elif nivel == 3:
            self.n_sec += 1
            self.n_sub = 0
            rotulo = f"{self.n_cap}.{self.n_sec}  {texto}"
            self.flow.append(CondPageBreak(3.4 * cm))
            self.flow.append(self._titulo_marcado(rotulo, "h2", 1, rotulo))
        else:
            self.n_sub += 1
            rotulo = f"{self.n_cap}.{self.n_sec}.{self.n_sub}  {texto}"
            self.flow.append(CondPageBreak(2.6 * cm))
            self.flow.append(Paragraph(formata_inline(rotulo, 11), E["h3"]))

    def _add_formula(self, latex: str) -> None:
        self.flow.append(Spacer(1, 5))
        self.flow.append(ImagemFormula(latex))
        self.flow.append(Spacer(1, 8))

    def _add_codigo(self, codigo: str, lang: str) -> None:
        codigo = codigo.rstrip("\n")
        linhas = [
            Paragraph(_realce(ln, lang) or "&nbsp;", E["codigo"])
            for ln in codigo.split("\n")
        ]
        self.flow.append(Spacer(1, 3))
        self.flow.append(_bloco_colorido(linhas, CODIGO_BG, CINZA_LINHA, pad=6))
        self.flow.append(Spacer(1, 9))

    def _add_caixa(self, linhas: list[str]) -> None:
        tipo, titulo = None, None
        if linhas and re.match(r"^\s*\[!\w+\]", linhas[0]):
            m = re.match(r"^\s*\[!(\w+)\]\s*(.*)$", linhas[0])
            tipo = m.group(1).upper()
            resto = m.group(2).strip()
            linhas = ([resto] if resto else []) + linhas[1:]
            titulo = CAIXAS.get(tipo, ("Nota", AZUL, AZUL_CLARO))[0]
        rotulo, cor, fundo = CAIXAS.get(tipo or "NOTA", ("Nota", AZUL, AZUL_CLARO))
        conteudo = []
        if tipo:
            conteudo.append(Paragraph(
                rotulo.upper(),
                ParagraphStyle("ct", parent=E["caixa_titulo"], textColor=cor)))
        # agrupa em paragrafos separados por linha em branco
        bloco: list[str] = []
        for ln in linhas + [""]:
            if ln.strip():
                bloco.append(ln.strip())
            elif bloco:
                conteudo.append(self._p(" ".join(bloco), "caixa"))
                bloco = []
        self.flow.append(Spacer(1, 3))
        self.flow.append(_bloco_colorido(conteudo, fundo, cor))
        self.flow.append(Spacer(1, 9))

    def _add_lista(self, linhas: list[str]) -> None:
        itens, atual, ordenada = [], None, False
        for ln in linhas:
            m = re.match(r"^(\s*)(?:([-*+])|(\d+)\.)\s+(.*)$", ln)
            if m:
                if atual is not None:
                    itens.append(atual)
                ordenada = m.group(3) is not None
                atual = m.group(4).strip()
            elif atual is not None:
                atual += " " + ln.strip()
        if atual is not None:
            itens.append(atual)
        flow_itens = [ListItem(self._p(t, "item"), leftIndent=14,
                               value=k + 1 if ordenada else None)
                      for k, t in enumerate(itens)]
        self.flow.append(ListFlowable(
            flow_itens, bulletType="1" if ordenada else "bullet",
            bulletFontName="DejaVu", bulletFontSize=9,
            bulletColor=AZUL, leftIndent=16, spaceBefore=2, spaceAfter=8,
            start=1 if ordenada else None))

    def _add_tabela(self, linhas: list[str]) -> None:
        def celulas(ln: str) -> list[str]:
            return [c.strip() for c in ln.strip().strip("|").split("|")]

        cab = celulas(linhas[0])
        corpo = [celulas(ln) for ln in linhas[2:] if ln.strip("| ")]
        ncols = len(cab)
        dados = [[Paragraph(formata_inline(c, 8.6), E["tabela_cab"]) for c in cab]]
        for linha in corpo:
            linha = (linha + [""] * ncols)[:ncols]
            dados.append([Paragraph(formata_inline(c, 8.6), E["tabela"])
                          for c in linha])
        larg = LARGURA_UTIL / ncols
        t = Table(dados, colWidths=[larg] * ncols, repeatRows=1, hAlign="CENTER")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), AZUL),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, AZUL_CLARO]),
            ("GRID", (0, 0), (-1, -1), 0.4, CINZA_LINHA),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, AZUL_ESCURO),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        self.flow.append(Spacer(1, 4))
        self.flow.append(t)
        self.flow.append(Spacer(1, 10))


# --------------------------------------------------------------------------
# Template do documento: capa, sumario, cabecalho e rodape
# --------------------------------------------------------------------------
MARGEM = 2.3 * cm
TOPO = 2.2 * cm
BASE = 2.0 * cm


class DocumentoCurso(BaseDocTemplate):
    def __init__(self, caminho: str, doc: Documento, **kw):
        super().__init__(
            caminho, pagesize=A4,
            leftMargin=MARGEM, rightMargin=MARGEM,
            topMargin=TOPO, bottomMargin=BASE,
            title=doc.titulo, author=doc.meta.get("autor", "Curso de ML & DL"),
            subject=doc.meta.get("tema", ""), **kw)
        self.doc_md = doc
        quadro = Frame(MARGEM, BASE, LARGURA_UTIL,
                       A4[1] - TOPO - BASE, id="corpo",
                       leftPadding=0, rightPadding=0,
                       topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id="capa", frames=[quadro], onPage=self._capa),
            PageTemplate(id="normal", frames=[quadro], onPage=self._moldura),
        ])

    # registra as entradas do sumario conforme os titulos sao desenhados
    def afterFlowable(self, flowable):
        info = getattr(flowable, "_toc_info", None)
        if info is not None:
            nivel, texto, _ = info
            self.notify("TOCEntry", (nivel, texto, self.page))

    def _capa(self, canv, doc):
        canv.saveState()
        canv.setFillColor(AZUL_ESCURO)
        canv.rect(0, A4[1] - 1.15 * cm, A4[0], 1.15 * cm, stroke=0, fill=1)
        canv.setFillColor(AZUL)
        canv.rect(0, 0, A4[0], 0.55 * cm, stroke=0, fill=1)
        canv.restoreState()

    def _moldura(self, canv, doc):
        canv.saveState()
        tema = self.doc_md.meta.get("tema", "")
        canv.setFont("DejaVu", 7.6)
        canv.setFillColor(CINZA_MEDIO)
        canv.drawString(MARGEM, A4[1] - TOPO + 0.62 * cm,
                        tema.upper()[:78])
        canv.drawRightString(A4[0] - MARGEM, A4[1] - TOPO + 0.62 * cm,
                             self.doc_md.titulo[:60])
        canv.setStrokeColor(CINZA_LINHA)
        canv.setLineWidth(0.5)
        canv.line(MARGEM, A4[1] - TOPO + 0.42 * cm,
                  A4[0] - MARGEM, A4[1] - TOPO + 0.42 * cm)
        canv.line(MARGEM, BASE - 0.65 * cm, A4[0] - MARGEM, BASE - 0.65 * cm)
        canv.setFont("DejaVu", 7.6)
        canv.drawString(MARGEM, BASE - 1.1 * cm,
                        "Machine Learning & Deep Learning — material do curso")
        canv.setFont("DejaVu-Bold", 8.4)
        canv.setFillColor(AZUL)
        canv.drawRightString(A4[0] - MARGEM, BASE - 1.12 * cm, str(canv.getPageNumber()))
        canv.restoreState()


def _capa_flow(doc: Documento) -> list:
    m = doc.meta
    f = [Spacer(1, 3.4 * cm)]
    if m.get("tema"):
        f.append(Paragraph(
            m["tema"].upper(),
            ParagraphStyle("t", fontName="DejaVu-Bold", fontSize=10,
                           textColor=AZUL, leading=14, spaceAfter=6)))
    f.append(HRFlowable(width="28%", thickness=2.4, color=AZUL_ESCURO,
                        hAlign="LEFT", spaceAfter=16))
    f.append(Paragraph(html.escape(doc.titulo), E["capa_titulo"]))
    if m.get("subtitulo"):
        f.append(Paragraph(formata_inline(m["subtitulo"], 13), E["capa_sub"]))
    f.append(Spacer(1, 1.6 * cm))
    if m.get("resumo"):
        f.append(_bloco_colorido(
            [Paragraph(formata_inline(m["resumo"], 9.6),
                       ParagraphStyle("r", parent=E["caixa"], fontSize=9.6))],
            AZUL_CLARO, AZUL))
    f.append(Spacer(1, 1.4 * cm))
    linhas_meta = []
    for chave, rotulo in (("nivel", "Nivel"), ("prerequisitos", "Pre-requisitos"),
                          ("duracao", "Carga sugerida"), ("notebooks", "Notebooks"),
                          ("autor", "Autoria"), ("versao", "Versao")):
        if m.get(chave):
            linhas_meta.append(f"<b>{rotulo}:</b> {html.escape(m[chave])}")
    if linhas_meta:
        f.append(Paragraph("<br/>".join(linhas_meta), E["capa_meta"]))
    return f


def _toc_flow() -> list:
    toc = TableOfContents()
    toc.levelStyles = [E["toc1"], E["toc2"]]
    toc.dotsMinLevel = 0
    return [
        Paragraph("Sumário", ParagraphStyle(
            "s", fontName="DejaVu-Bold", fontSize=16, leading=21,
            textColor=AZUL_ESCURO, spaceAfter=4)),
        HRFlowable(width="100%", thickness=1.2, color=AZUL, spaceAfter=12),
        toc,
    ]


def constroi_pdf(caminho_md: str | Path, caminho_pdf: str | Path | None = None) -> Path:
    caminho_md = Path(caminho_md)
    texto = caminho_md.read_text(encoding="utf-8")
    doc = Documento(texto, str(caminho_md))
    destino = Path(caminho_pdf) if caminho_pdf else caminho_md.with_suffix(".pdf")

    modelo = DocumentoCurso(str(destino), doc)
    historia = _capa_flow(doc)
    historia.append(NextPageTemplate("normal"))
    historia.append(PageBreak())
    historia += _toc_flow()
    historia.append(PageBreak())
    historia += doc.flow
    modelo.multiBuild(historia)
    return destino


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("uso: md2pdf.py <arquivo.md> [...]  |  md2pdf.py --todos", file=sys.stderr)
        return 2
    alvos: list[Path] = []
    if argv[1] == "--todos":
        alvos = sorted(RAIZ.rglob("teoria*.md"))
    else:
        alvos = [Path(a) for a in argv[1:]]
    for md in alvos:
        try:
            pdf = constroi_pdf(md)
            kb = pdf.stat().st_size / 1024
            try:
                rotulo = pdf.relative_to(RAIZ)
            except ValueError:
                rotulo = pdf
            print(f"  OK  {rotulo}  ({kb:,.0f} KB)")
        except ErroDeFormula as exc:
            print(f"  ERRO DE FORMULA em {md}:\n{exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERRO em {md}: {type(exc).__name__}: {exc}", file=sys.stderr)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
