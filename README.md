# Cribado No Invasivo de Hígado Graso (MASLD)

Herramienta de cribado de esteatosis hepática (MASLD) para la consulta nutricional, 
desarrollada como Trabajo Fin de Máster (Data Science, Big Data & Business Analytics — UCM).

Estima el grado de esteatosis hepática (parámetro **CAP** de la elastografía por FibroScan) 
a partir de variables clínicas accesibles en consulta, sin necesidad del propio FibroScan.

## Aplicación en línea

**Probar la herramienta:** https://cribado-masld-fycc5vku6nsadxrsv5ae88.streamlit.app

## Descripción del proyecto

El hígado graso es la enfermedad hepática más prevalente del mundo y está masivamente 
infradiagnosticada (en la muestra analizada, el **95% de los casos no tenían diagnóstico previo**). 
Su detección objetiva requiere una elastografía por FibroScan, un equipo poco accesible en la 
consulta nutricional.

Este proyecto entrena un modelo de *machine learning* sobre ~12.500 adultos de la encuesta 
**NHANES** para estimar el CAP y clasificar el riesgo de esteatosis, y lo despliega como 
aplicación web funcional.

**Resultados principales:**
- Cribado no invasivo (cintura, talla, edad, sexo): **AUROC 0,81**
- Con analítica de sangre (+ TyG, HbA1c, HDL): **AUROC 0,83**
- Hallazgo central: la adiposidad central domina la predicción; la dieta autorreportada no aporta.

## Contenido del repositorio

- `TFM.ipynb` — análisis completo, modelado e interpretabilidad
- `app_deploy.py` — aplicación web (Streamlit)
- `modelos_app.pkl` — modelos entrenados
- `requirements.txt` — dependencias

##  Reproducibilidad

Los datos provienen de **NHANES** (CDC), de acceso público y sin restricciones de uso, 
descargables desde https://www.cdc.gov/nchs/nhanes/

Se emplearon los ciclos **2017-2020** (prefijo `P_`) y **2021-2023** (sufijo `_L`), tomando 
los módulos: elastografía (LUX), demografía (DEMO), bioquímica (BIOPRO), HbA1c (GHB), 
HDL, dieta (DR1TOT/DR2TOT), antropometría (BMX), presión (BPXO), y los cuestionarios de 
alcohol (ALQ), actividad física (PAQ), sueño (SLQ), salud mental (DPQ), condiciones médicas 
(MCQ) y hepatitis (HEPC/HEPBD).

El notebook `TFM.ipynb` documenta el proceso completo: construcción de la cohorte, ingeniería 
de variables, modelado, interpretabilidad (SHAP) y validación.

## 🛠️ Tecnologías

Python · pandas · scikit-learn · LightGBM · SHAP · Streamlit · Plotly

## 👤 Autor

**Lic. Bruno Inuggi** — Nutricionista y estudiante del Máster en Data Science (UCM)  
GitHub: [@BrunoInuggi](https://github.com/BrunoInuggi)
