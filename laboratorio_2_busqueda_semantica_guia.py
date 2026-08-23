"""
CC3103 - Procesamiento de Lenguaje Natural
Laboratorio 2: Motor de Busqueda Semantica

GUIA PARA ESTUDIANTES

Objetivos:
- Generar embeddings de oraciones usando un modelo local de Hugging Face.
- Calcular similitud coseno entre una consulta y un corpus.
- Devolver resultados top-k.
- Comparar busqueda semantica contra busqueda por palabras clave.

Instalacion requerida:
    pip install sentence-transformers numpy scikit-learn

Ejecutar:
    python laboratorio_2_busqueda_semantica_guia.py
"""

from __future__ import annotations

from typing import Iterable
import re

import numpy as np


MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# -----------------------------------------------------------------------------
# 1. Corpus De Prueba
# -----------------------------------------------------------------------------

# TODO: Modifique o amplie este corpus para incluir al menos 20 oraciones.
# Recomendacion: use un dominio concreto, por ejemplo soporte tecnico,
# biblioteca universitaria, servicios bancarios, salud o comercio electronico.
CORPUS = [
    "No puedo iniciar sesion en mi cuenta.",
    "Olvide mi clave de acceso al sistema.",
    "El usuario desea cambiar su contrasena.",
    "La plataforma muestra un error al iniciar sesion.",
    "El cliente quiere actualizar su correo electronico.",
    "La factura fue enviada al correo registrado.",
    "El pedido llegara manana por la tarde.",
    "El sistema recomienda restablecer la clave.",
    "La aplicacion movil no carga correctamente.",
    "El clima estara lluvioso durante la tarde.",
]


# TODO: Agregue al menos 5 consultas de prueba.
CONSULTAS = [
    "problemas con mi contrasena",
    "no puedo entrar a mi cuenta",
    "quiero cambiar mi email",
]


# -----------------------------------------------------------------------------
# 2. Utilidades De Instalacion
# -----------------------------------------------------------------------------

def cargar_modelo(nombre_modelo: str):
    """Carga un modelo de sentence-transformers.

    Si la dependencia no esta instalada, muestra una instruccion clara.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        mensaje = (
            "No se encontro la libreria 'sentence-transformers'.\n"
            "Instale las dependencias con:\n\n"
            "    pip install sentence-transformers numpy scikit-learn\n\n"
            "Si usa Google Colab:\n\n"
            "    !pip install sentence-transformers numpy scikit-learn\n"
        )
        raise SystemExit(mensaje) from exc

    return SentenceTransformer(nombre_modelo)


# -----------------------------------------------------------------------------
# 3. Embeddings
# -----------------------------------------------------------------------------

def generar_embeddings(modelo, textos: list[str]) -> np.ndarray:
    """Genera embeddings para una lista de textos.

    normalize_embeddings=True permite que el producto punto sea equivalente
    a similitud coseno para muchos usos practicos.
    """
    embeddings = modelo.encode(textos, normalize_embeddings=True)
    return np.asarray(embeddings)


# -----------------------------------------------------------------------------
# 4. Similitud Coseno Y Ranking
# -----------------------------------------------------------------------------

def similitud_coseno(embedding_consulta: np.ndarray, embeddings_corpus: np.ndarray) -> np.ndarray:
    """Calcula similitud coseno entre una consulta y todos los documentos.

    Como los embeddings estan normalizados, el producto punto equivale a la
    similitud coseno.
    """
    return embeddings_corpus @ embedding_consulta


def buscar_semanticamente(
    consulta: str,
    corpus: list[str],
    embeddings_corpus: np.ndarray,
    modelo,
    top_k: int = 3,
) -> list[dict]:
    """Devuelve los top-k textos semanticamente mas similares a la consulta."""
    embedding_consulta = generar_embeddings(modelo, [consulta])[0]
    puntajes = similitud_coseno(embedding_consulta, embeddings_corpus)

    indices_ordenados = np.argsort(puntajes)[::-1][:top_k]

    resultados = []
    for posicion, indice in enumerate(indices_ordenados, start=1):
        resultados.append({
            "rank": posicion,
            "indice": int(indice),
            "texto": corpus[indice],
            "score": float(puntajes[indice]),
        })

    return resultados


# -----------------------------------------------------------------------------
# 5. Busqueda Por Palabras Clave
# -----------------------------------------------------------------------------

def tokenizar_simple(texto: str) -> set[str]:
    """Tokeniza texto para una busqueda por palabras clave simple."""
    return set(re.findall(r"\b\w+\b", texto.lower()))


def buscar_por_palabras_clave(consulta: str, corpus: list[str], top_k: int = 3) -> list[dict]:
    """Busca textos por cantidad de palabras compartidas con la consulta.

    Esta implementacion es intencionalmente simple para compararla contra
    busqueda semantica.
    """
    tokens_consulta = tokenizar_simple(consulta)
    resultados = []

    for indice, texto in enumerate(corpus):
        tokens_texto = tokenizar_simple(texto)
        coincidencias = tokens_consulta.intersection(tokens_texto)
        score = len(coincidencias)

        resultados.append({
            "indice": indice,
            "texto": texto,
            "score": score,
            "coincidencias": sorted(coincidencias),
        })

    resultados.sort(key=lambda item: item["score"], reverse=True)
    return resultados[:top_k]


# -----------------------------------------------------------------------------
# 6. Impresion De Resultados
# -----------------------------------------------------------------------------

def imprimir_resultados_semanticos(consulta: str, resultados: Iterable[dict]) -> None:
    print("\nBusqueda semantica")
    print(f"Consulta: {consulta}")

    for resultado in resultados:
        print(
            f"  {resultado['rank']}. "
            f"score={resultado['score']:.4f} | "
            f"{resultado['texto']}"
        )


def imprimir_resultados_keyword(consulta: str, resultados: Iterable[dict]) -> None:
    print("\nBusqueda por palabras clave")
    print(f"Consulta: {consulta}")

    for posicion, resultado in enumerate(resultados, start=1):
        coincidencias = ", ".join(resultado["coincidencias"]) or "sin coincidencias"
        print(
            f"  {posicion}. "
            f"score={resultado['score']} | "
            f"coincidencias={coincidencias} | "
            f"{resultado['texto']}"
        )


def comparar_busquedas(
    consulta: str,
    corpus: list[str],
    embeddings_corpus: np.ndarray,
    modelo,
    top_k: int = 3,
) -> None:
    """Imprime busqueda semantica y keyword search para una consulta."""
    print("=" * 100)
    print(f"CONSULTA: {consulta}")
    print("=" * 100)

    resultados_semanticos = buscar_semanticamente(
        consulta=consulta,
        corpus=corpus,
        embeddings_corpus=embeddings_corpus,
        modelo=modelo,
        top_k=top_k,
    )
    resultados_keyword = buscar_por_palabras_clave(consulta, corpus, top_k=top_k)

    imprimir_resultados_semanticos(consulta, resultados_semanticos)
    imprimir_resultados_keyword(consulta, resultados_keyword)
    print()


# -----------------------------------------------------------------------------
# 7. Ejercicios Para Estudiantes
# -----------------------------------------------------------------------------

def ejercicio_1_ampliar_corpus():
    """TODO para estudiantes.

    Amplie CORPUS hasta tener al menos 20 oraciones.
    Use un dominio claro y coherente.
    """
    pass


def ejercicio_2_agregar_consultas():
    """TODO para estudiantes.

    Agregue al menos 5 consultas.
    Incluya consultas que no compartan palabras exactas con la respuesta esperada.
    """
    pass


def ejercicio_3_analizar_resultados():
    """TODO para estudiantes.

    En su informe, responda:
    1. En que consulta funciono mejor la busqueda semantica?
    2. En que consulta funciono mejor keyword search?
    3. Que resultado fue inesperado?
    4. Que cambiaria del corpus para mejorar la busqueda?
    """
    pass


# -----------------------------------------------------------------------------
# 8. Programa Principal
# -----------------------------------------------------------------------------

def main() -> None:
    print("Cargando modelo de embeddings...")
    modelo = cargar_modelo(MODELO_EMBEDDINGS)

    print("Generando embeddings del corpus...")
    embeddings_corpus = generar_embeddings(modelo, CORPUS)

    print(f"Corpus: {len(CORPUS)} oraciones")
    print(f"Dimension de embeddings: {embeddings_corpus.shape[1]}")

    for consulta in CONSULTAS:
        comparar_busquedas(
            consulta=consulta,
            corpus=CORPUS,
            embeddings_corpus=embeddings_corpus,
            modelo=modelo,
            top_k=3,
        )


if __name__ == "__main__":
    main()
