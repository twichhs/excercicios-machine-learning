"""
py2nb.py — Converte scripts no formato "percent" em notebooks .ipynb.

Por que escrever os notebooks como .py e nao direto em JSON? Porque o .py e
legivel, versiona bem no git, e — principalmente — pode ser EXECUTADO como um
script comum antes de virar notebook. Assim nenhum notebook do curso chega ao
aluno com codigo quebrado: o build so gera o .ipynb se o .py rodar inteiro.

Formato de entrada:

    # %% [markdown]
    # # Titulo em Markdown
    # Texto normal, uma linha por linha, todas prefixadas por "# ".

    # %%
    codigo_python_normal = "sem prefixo"

Uso:
    python py2nb.py caminho/para/notebook_fonte.py         # gera + executa
    python py2nb.py --sem-executar arquivo.py              # so converte
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

KERNEL = {
    "display_name": "Python (curso ML)",
    "language": "python",
    "name": "curso-ml",
}

_RE_CELULA = re.compile(r"^#\s*%%\s*(\[markdown\])?\s*$")


def divide_celulas(texto: str) -> list[tuple[str, str]]:
    """Quebra o script em [(tipo, conteudo)], tipo em {'markdown', 'code'}."""
    celulas: list[tuple[str, str]] = []
    tipo_atual: str | None = None
    buffer: list[str] = []

    def fecha() -> None:
        if tipo_atual is None:
            return
        corpo = "\n".join(buffer).strip("\n")
        if not corpo.strip():
            return
        if tipo_atual == "markdown":
            # remove o "# " que prefixa cada linha de markdown
            corpo = "\n".join(
                re.sub(r"^#\s?", "", ln) for ln in corpo.split("\n"))
        celulas.append((tipo_atual, corpo))

    for linha in texto.replace("\r\n", "\n").split("\n"):
        m = _RE_CELULA.match(linha.rstrip())
        if m:
            fecha()
            tipo_atual = "markdown" if m.group(1) else "code"
            buffer = []
        else:
            buffer.append(linha)
    fecha()
    return celulas


def converte(origem: Path, destino: Path | None = None) -> Path:
    destino = destino or origem.with_suffix(".ipynb")
    nb = new_notebook()
    for tipo, corpo in divide_celulas(origem.read_text(encoding="utf-8")):
        nb.cells.append(
            new_markdown_cell(corpo) if tipo == "markdown" else new_code_cell(corpo))
    nb.metadata["kernelspec"] = KERNEL
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
    nbformat.write(nb, str(destino))
    return destino


def executa(caminho_nb: Path, timeout: int = 900) -> None:
    """Roda o notebook e grava as saidas dentro dele."""
    from nbclient import NotebookClient

    nb = nbformat.read(str(caminho_nb), as_version=4)
    cliente = NotebookClient(
        nb, timeout=timeout, kernel_name="curso-ml",
        resources={"metadata": {"path": str(caminho_nb.parent)}})
    cliente.execute()
    nbformat.write(nb, str(caminho_nb))


def main(argv: list[str]) -> int:
    sem_exec = "--sem-executar" in argv
    args = [a for a in argv[1:] if not a.startswith("--")]
    # forma "py2nb.py fonte.py destino.ipynb": converte um so, com nome dado
    saida: Path | None = None
    if len(args) == 2 and args[1].endswith(".ipynb"):
        args, saida = args[:1], Path(args[1])
    alvos = [Path(a) for a in args]
    if not alvos:
        print("uso: py2nb.py [--sem-executar] <fonte.py> [destino.ipynb]",
              file=sys.stderr)
        return 2
    for py in alvos:
        nb = converte(py, saida)
        if sem_exec:
            print(f"  convertido  {nb.name}")
            continue
        try:
            executa(nb)
        except Exception as exc:  # noqa: BLE001
            print(f"  FALHOU  {nb}\n     {type(exc).__name__}: "
                  f"{str(exc)[:600]}", file=sys.stderr)
            return 1
        print(f"  OK  {nb.name}  ({len(nbformat.read(str(nb), as_version=4).cells)} células)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
