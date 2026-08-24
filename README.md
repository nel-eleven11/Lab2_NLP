# Lab2_NLP — Motor de búsqueda semántica

Laboratorio 2 de Procesamiento de Lenguaje Natural (CC3103). Construye un motor de
búsqueda semántica simple con embeddings locales de Hugging Face y lo compara contra una
búsqueda por palabras clave sobre un corpus de soporte técnico en español.

## Contenido

- `laboratorio_2_busqueda_semantica.ipynb` — solución completa (Partes A–G + reflexión).
- `laboratorio_2_busqueda_semantica_guia.py` — guía base provista.

## Pipeline

Corpus → embeddings → consulta → embedding de la consulta → similitud coseno →
ranking top-k → comparación con keyword search.

El notebook cubre:

- **A/B** — corpus de 24 oraciones y 6 consultas de prueba (secciones 7–8).
- **C** — generación de embeddings con `paraphrase-multilingual-MiniLM-L12-v2` (sección 9).
- **D** — similitud coseno vía producto punto sobre vectores normalizados (sección 10).
- **E** — recuperación top-k (sección 11).
- **F** — búsqueda por palabras clave como línea base (sección 12).
- **G** — comparación semántica vs. léxica para cada consulta (sección 13).
- Reflexión de 150–250 palabras (sección 14).

## Dependencias

```bash
pip install sentence-transformers numpy scikit-learn
```

## Ejecución

```bash
jupyter notebook laboratorio_2_busqueda_semantica.ipynb
```

Ejecute las celdas en orden. La primera ejecución descarga el modelo de embeddings.
