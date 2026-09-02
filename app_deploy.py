import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go

st.set_page_config(page_title='Cribado MASLD', page_icon='🩺', layout='centered')

st.markdown('''
<style>
.stApp { background: linear-gradient(160deg, #e8f5e9 0%, #f1f8f4 45%, #ffffff 100%); }
.stApp, .stApp p, .stApp label, .stApp span, .stApp div,
.stMarkdown, [data-testid="stWidgetLabel"] { color: #1a1a1a !important; }
h1 { color: #1b5e20 !important; font-weight: 700; }
h2, h3, h4, h5 { color: #2e7d32 !important; }
.stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important; color: #1a1a1a !important; }
.ref-nota { font-size: 0.75rem; color: #6b6b6b !important; margin-top: -8px; margin-bottom: 6px; }
.stButton>button {
    background: linear-gradient(90deg, #2e7d32, #43a047);
    color: white !important; border: none; border-radius: 10px;
    padding: 0.55rem 1.6rem; font-weight: 600; font-size: 1rem; }
.stButton>button:hover { background: linear-gradient(90deg, #1b5e20, #388e3c); }
[data-testid="stMetricValue"] { color: #1b5e20 !important; }
[data-testid="stMetricLabel"] { color: #1a1a1a !important; }
</style>
''', unsafe_allow_html=True)

m = joblib.load('modelos_app.pkl')
modelo_simple = m['modelo_simple']; vars_simple = m['vars_simple']
modelo_sangre = m['modelo_sangre']; vars_sangre = m['vars_sangre']

UMBRAL_CRIBADO = 265
nombres = {'WHtR': 'Cintura/talla (WHtR)', 'RIDAGEYR': 'Edad', 'RIAGENDR': 'Sexo',
           'TyG': 'Resistencia insulínica (TyG)', 'HbA1c': 'HbA1c', 'HDL': 'HDL colesterol'}

def ref(texto):
    st.markdown(f'<div class="ref-nota">{texto}</div>', unsafe_allow_html=True)

def dibujar_gauge(cap):
    if cap < 248: color = '#2e7d32'; grado = 'S0 — Sin esteatosis'
    elif cap < 268: color = '#f9a825'; grado = 'S1 — Leve'
    elif cap < 280: color = '#ef6c00'; grado = 'S2 — Moderada'
    else: color = '#c62828'; grado = 'S3 — Severa'
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=cap,
        number={'suffix': ' dB/m', 'font': {'size': 40, 'color': '#1a1a1a'}},
        title={'text': f'CAP estimado — {grado}', 'font': {'size': 18, 'color': '#2e7d32'}},
        gauge={'axis': {'range': [100, 400], 'tickcolor': '#1a1a1a', 'tickfont': {'color': '#1a1a1a'}},
               'bar': {'color': color, 'thickness': 0.3},
               'steps': [{'range': [100, 248], 'color': '#e8f5e9'},
                         {'range': [248, 268], 'color': '#fff8e1'},
                         {'range': [268, 280], 'color': '#ffe0b2'},
                         {'range': [280, 400], 'color': '#ffcdd2'}],
               'threshold': {'line': {'color': '#1565c0', 'width': 4},
                             'thickness': 0.8, 'value': UMBRAL_CRIBADO}}))
    fig.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30),
                      paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#1a1a1a'))
    return fig

st.title('🩺 Cribado MASLD')
st.markdown('##### Estimación no invasiva del hígado graso desde la consulta')
st.write('')

modo = st.radio('¿Qué datos tiene disponibles del paciente?',
                ['Solo medidas (sin analítica)', 'Con analítica de sangre'])

st.header('Datos del paciente')
col1, col2 = st.columns(2)
with col1:
    cintura = st.number_input('Cintura (cm)', min_value=0.0, max_value=200.0, value=0.0)
    edad = st.number_input('Edad (años)', min_value=0, max_value=100, value=0)
with col2:
    talla = st.number_input('Talla (cm)', min_value=0.0, max_value=220.0, value=0.0)
    sexo = st.selectbox('Sexo', options=['— Seleccionar —', 'Hombre', 'Mujer'])

if modo == 'Con analítica de sangre':
    st.subheader('Analítica de sangre')
    col3, col4 = st.columns(2)
    with col3:
        trigliceridos = st.number_input('Triglicéridos (mg/dL)', min_value=0.0, max_value=1000.0, value=0.0)
        ref('Ref.: <150')
        hba1c = st.number_input('HbA1c (%)', min_value=0.0, max_value=15.0, value=0.0)
        ref('Ref.: <5,7')
    with col4:
        glucosa = st.number_input('Glucosa (mg/dL)', min_value=0.0, max_value=500.0, value=0.0)
        ref('Ref.: 70–99')
        hdl = st.number_input('HDL colesterol (mg/dL)', min_value=0.0, max_value=150.0, value=0.0)
        ref('Ref.: >40 (H) / >50 (M)')

    st.subheader('Datos para FIB-4 (fibrosis) — opcional')
    st.caption('Si dispone de estos valores, se calcula el índice FIB-4 de fibrosis hepática.')
    col5, col6, col7 = st.columns(3)
    with col5:
        ast = st.number_input('AST/GOT (U/L)', min_value=0.0, max_value=500.0, value=0.0)
        ref('Ref.: 10–40')
    with col6:
        alt = st.number_input('ALT/GPT (U/L)', min_value=0.0, max_value=500.0, value=0.0)
        ref('Ref.: 7–56')
    with col7:
        plaquetas = st.number_input('Plaquetas (10⁹/L)', min_value=0.0, max_value=800.0, value=0.0)
        ref('Ref.: 150–400')

st.write('')
if st.button('Estimar'):
    faltan = []
    if cintura <= 0: faltan.append('cintura')
    if talla <= 0: faltan.append('talla')
    if edad <= 0: faltan.append('edad')
    if sexo == '— Seleccionar —': faltan.append('sexo')
    if modo == 'Con analítica de sangre':
        if trigliceridos <= 0: faltan.append('triglicéridos')
        if glucosa <= 0: faltan.append('glucosa')
        if hba1c <= 0: faltan.append('HbA1c')
        if hdl <= 0: faltan.append('HDL')
    if faltan:
        st.error(f'⚠️ Faltan datos obligatorios: {", ".join(faltan)}. Complete todos los campos.')
        st.session_state['estimado'] = False
    else:
        st.session_state['estimado'] = True
        st.session_state['modo'] = modo
        st.session_state['datos'] = dict(cintura=cintura, talla=talla, edad=edad, sexo=sexo)
        if modo == 'Con analítica de sangre':
            st.session_state['datos'].update(dict(trigliceridos=trigliceridos, glucosa=glucosa,
                                                  hba1c=hba1c, hdl=hdl, ast=ast, alt=alt, plaquetas=plaquetas))

if st.session_state.get('estimado', False):
    d = st.session_state['datos']
    modo_g = st.session_state['modo']
    whtr = d['cintura'] / d['talla']
    sexo_cod = 1 if d['sexo'] == 'Hombre' else 2

    if modo_g == 'Solo medidas (sin analítica)':
        X = pd.DataFrame([[whtr, d['edad'], sexo_cod]], columns=vars_simple)
        modelo_usado = modelo_simple
    else:
        tyg = np.log((d['trigliceridos'] * d['glucosa']) / 2)
        X = pd.DataFrame([[whtr, d['edad'], sexo_cod, tyg, d['hba1c'], d['hdl']]], columns=vars_sangre)
        modelo_usado = modelo_sangre
    cap = modelo_usado.predict(X)[0]

    st.header('Resultado — Esteatosis')
    st.plotly_chart(dibujar_gauge(cap), use_container_width=True)

    st.subheader('Cribado')
    if cap >= UMBRAL_CRIBADO:
        if modo_g == 'Solo medidas (sin analítica)':
            st.warning('⚠️ Riesgo de esteatosis — se sugiere completar con analítica de sangre para afinar la estimación.')
        else:
            st.error('⚠️ Riesgo de esteatosis — se sugiere valoración complementaria (elastografía).')
    else:
        if modo_g == 'Solo medidas (sin analítica)':
            st.success('✓ Bajo riesgo. Una analítica de sangre permitiría una estimación más precisa.')
        else:
            st.success('✓ Bajo riesgo de esteatosis en este cribado.')

    st.subheader('¿Qué factores influyen en este resultado?')
    explainer = shap.TreeExplainer(modelo_usado)
    shap_vals = explainer.shap_values(X)[0]
    contrib = pd.DataFrame({'factor': [nombres.get(c, c) for c in X.columns], 'impacto': shap_vals}
                          ).sort_values('impacto', key=abs, ascending=True)
    fig_b = go.Figure(go.Bar(x=contrib['impacto'], y=contrib['factor'], orientation='h', width=0.4,
                             marker_color=['#c62828' if v > 0 else '#2e7d32' for v in contrib['impacto']],
                             text=[f'{v:+.0f}' for v in contrib['impacto']], textposition='outside',
                             textfont=dict(color='#1a1a1a')))
    fig_b.update_layout(
        xaxis=dict(title=dict(text='Impacto en el CAP (dB/m)', font=dict(color='#1a1a1a')),
                   tickfont=dict(color='#1a1a1a'), gridcolor='#d0d0d0', zerolinecolor='#999'),
        yaxis=dict(tickfont=dict(color='#1a1a1a', size=13)),
        height=240, margin=dict(t=10, b=40, l=10, r=30), bargap=0.15,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a1a1a'), showlegend=False)
    st.plotly_chart(fig_b, use_container_width=True)
    st.caption('🔴 Rojo: eleva el CAP.  🟢 Verde: lo reduce.')

    if modo_g == 'Con analítica de sangre' and d.get('ast',0) > 0 and d.get('alt',0) > 0 and d.get('plaquetas',0) > 0:
        fib4 = (d['edad'] * d['ast']) / (d['plaquetas'] * np.sqrt(d['alt']))
        corte_bajo = 2.0 if d['edad'] >= 65 else 1.3
        st.header('Resultado — Fibrosis (FIB-4)')
        st.metric('Índice FIB-4', f'{fib4:.2f}')
        if fib4 < corte_bajo: st.success('🟢 Fibrosis avanzada improbable')
        elif fib4 <= 2.67: st.warning('🟡 Zona indeterminada — considerar valoración')
        else: st.error('🔴 Fibrosis avanzada probable — derivar a especialista')
        st.caption(f'FIB-4 = (Edad × AST) / (Plaquetas × √ALT). Corte bajo: {corte_bajo} (ajustado por edad ≥65). Alto: >2,67.')

    st.write('---')
    st.header('🔄 Simulador de intervención')
    st.caption('Estime cómo cambiaría el CAP con la pérdida de peso (≈0,9 cm de cintura por kg). '
               'Comparación orientativa con perfiles similares, no una predicción individual garantizada.')

    def grado_cap(v):
        if v < 248: return 'S0 (sin esteatosis)'
        elif v < 268: return 'S1 (leve)'
        elif v < 280: return 'S2 (moderada)'
        else: return 'S3 (severa)'

    def predecir_con_kg(kg):
        c = max(d['cintura'] - kg * 0.9, 40.0)
        w = c / d['talla']
        if modo_g == 'Solo medidas (sin analítica)':
            Xx = pd.DataFrame([[w, d['edad'], sexo_cod]], columns=vars_simple)
        else:
            tg = np.log((d['trigliceridos'] * d['glucosa']) / 2)
            Xx = pd.DataFrame([[w, d['edad'], sexo_cod, tg, d['hba1c'], d['hdl']]], columns=vars_sangre)
        return modelo_usado.predict(Xx)[0], c

    kg_perdidos = st.slider('Kilos a perder', 0.0, 20.0, 5.0, 0.5)
    cap_sim, sim_cintura = predecir_con_kg(kg_perdidos)
    delta = cap_sim - cap

    if cap_sim < 248: color_s = '#2e7d32'
    elif cap_sim < 268: color_s = '#f9a825'
    elif cap_sim < 280: color_s = '#ef6c00'
    else: color_s = '#c62828'

    fig_s = go.Figure(go.Indicator(
        mode='gauge+number+delta', value=cap_sim,
        delta={'reference': cap, 'increasing': {'color': '#c62828'},
               'decreasing': {'color': '#2e7d32'}, 'suffix': ' dB/m'},
        number={'suffix': ' dB/m', 'font': {'size': 36, 'color': '#1a1a1a'}},
        title={'text': f'CAP proyectado tras perder {kg_perdidos:.1f} kg', 'font': {'size': 16, 'color': '#2e7d32'}},
        gauge={'axis': {'range': [100, 400], 'tickcolor': '#1a1a1a', 'tickfont': {'color': '#1a1a1a'}},
               'bar': {'color': color_s, 'thickness': 0.3},
               'steps': [{'range': [100, 248], 'color': '#e8f5e9'},
                         {'range': [248, 268], 'color': '#fff8e1'},
                         {'range': [268, 280], 'color': '#ffe0b2'},
                         {'range': [280, 400], 'color': '#ffcdd2'}],
               'threshold': {'line': {'color': '#1565c0', 'width': 4}, 'thickness': 0.8, 'value': cap}}))
    fig_s.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30),
                        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#1a1a1a'))
    st.plotly_chart(fig_s, use_container_width=True)
    st.caption(f'🔵 Línea azul = CAP actual ({cap:.0f} dB/m).  Cintura estimada: {d["cintura"]:.0f} → {sim_cintura:.0f} cm.')

    grado_actual = grado_cap(cap); grado_nuevo = grado_cap(cap_sim)
    if kg_perdidos > 0 and grado_nuevo != grado_actual:
        st.success(f'🎯 Con esta pérdida de peso, pasaría de **{grado_actual}** a **{grado_nuevo}**.')
    elif kg_perdidos > 0 and delta < 0:
        st.info(f'📉 La estimación baja {abs(delta):.0f} dB/m (se mantiene en grado {grado_nuevo}).')

    if cap >= UMBRAL_CRIBADO:
        kg_meta = None
        for kg in np.arange(0.5, 20.5, 0.5):
            ct, _ = predecir_con_kg(kg)
            if ct < UMBRAL_CRIBADO:
                kg_meta = kg; break
        if kg_meta:
            st.warning(f'🏁 Meta: perdiendo aproximadamente **{kg_meta:.1f} kg**, saldría de la zona de riesgo del cribado.')
        else:
            st.caption('Con pérdida de peso sola no se alcanza a salir de la zona de riesgo; '
                       'la intervención integral (dieta, ejercicio, metabolismo) es clave.')

    st.write('')
    st.caption('Herramienta de cribado orientativo, no diagnóstica. No sustituye la evaluación médica.')
