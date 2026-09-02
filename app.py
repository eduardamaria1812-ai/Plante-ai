import streamlit as st
from PIL import Image
import numpy as np
import cv2

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="PlantAI - Identificador de Plantas",
    page_icon="🌱",
    layout="wide"
)

# ============================================================
# BANCO DE DADOS DAS PLANTAS
# ============================================================

PLANTAS = {
    "Acerola": {
        "nome_cientifico": "Malpighia emarginata",
        "tipo": "Frutífera",
        "naturalidade": "Nativa da América Central e norte da América do Sul",
        "cores": ["verde", "vermelho", "rosa"],
        "caracteristicas": [
            "Folhas pequenas e verdes",
            "Frutos pequenos e arredondados",
            "Frutos geralmente vermelhos quando maduros",
            "Arbusto de pequeno a médio porte"
        ]
    },

    "Ipê-amarelo": {
        "nome_cientifico": "Handroanthus spp.",
        "tipo": "Árvore ornamental",
        "naturalidade": "Nativa do Brasil",
        "cores": ["verde", "amarelo"],
        "caracteristicas": [
            "Flores amarelas muito vistosas",
            "Folhas geralmente compostas",
            "Pode perder grande parte das folhas durante a floração",
            "Árvore de médio a grande porte"
        ]
    },

    "Ipê-roxo": {
        "nome_cientifico": "Handroanthus spp.",
        "tipo": "Árvore ornamental",
        "naturalidade": "Nativa do Brasil",
        "cores": ["verde", "roxo", "rosa"],
        "caracteristicas": [
            "Flores roxas ou rosadas",
            "Floração bastante chamativa",
            "Pode florescer com poucas folhas",
            "Árvore de médio a grande porte"
        ]
    },

    "Lajea rubra": {
        "nome_cientifico": "Lagerstroemia indica",
        "tipo": "Árvore ornamental",
        "naturalidade": "Originária da Ásia",
        "cores": ["verde", "vermelho", "rosa", "roxo"],
        "caracteristicas": [
            "Flores agrupadas em cachos",
            "Flores podem apresentar tonalidades vermelhas, rosas ou roxas",
            "Folhas verdes",
            "Muito utilizada na arborização urbana"
        ]
    },

    "Magnólia": {
        "nome_cientifico": "Magnolia spp.",
        "tipo": "Árvore ornamental",
        "naturalidade": "Originária principalmente da Ásia e América",
        "cores": ["verde", "branco", "rosa", "roxo"],
        "caracteristicas": [
            "Flores grandes e vistosas",
            "Folhas geralmente largas",
            "Flores podem ser brancas, rosas ou arroxeadas",
            "Pode atingir grande porte"
        ]
    },

    "Palmeira-imperial": {
        "nome_cientifico": "Roystonea oleracea",
        "tipo": "Palmeira ornamental",
        "naturalidade": "Nativa do Caribe e norte da América do Sul",
        "cores": ["verde", "marrom"],
        "caracteristicas": [
            "Tronco alto, liso e geralmente acinzentado",
            "Copa formada por grandes folhas",
            "Formato característico de palmeira",
            "Pode atingir grande altura"
        ]
    },

    "Sibipiruna": {
        "nome_cientifico": "Cenostigma pluviosum",
        "tipo": "Árvore ornamental",
        "naturalidade": "Nativa do Brasil",
        "cores": ["verde", "amarelo"],
        "caracteristicas": [
            "Folhas compostas",
            "Flores amarelas",
            "Copa ampla e arredondada",
            "Muito utilizada na arborização urbana"
        ]
    }
}

# ============================================================
# FUNÇÃO PARA ANALISAR A IMAGEM
# ============================================================

def analisar_imagem(imagem):

    img = np.array(imagem)

    # RGB -> HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # --------------------------------------------------------
    # IDENTIFICAÇÃO DE CORES
    # --------------------------------------------------------

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    total = h.size

    # Verde
    verde = np.sum(
        (h >= 35) &
        (h <= 90) &
        (s > 60) &
        (v > 40)
    ) / total

    # Amarelo
    amarelo = np.sum(
        (h >= 15) &
        (h <= 35) &
        (s > 70) &
        (v > 80)
    ) / total

    # Vermelho
    vermelho = np.sum(
        ((h <= 10) | (h >= 170)) &
        (s > 70) &
        (v > 60)
    ) / total

    # Roxo
    roxo = np.sum(
        (h >= 125) &
        (h <= 165) &
        (s > 50)
    ) / total

    # Rosa
    rosa = np.sum(
        (h >= 160) &
        (h <= 175) &
        (s > 40)
    ) / total

    # --------------------------------------------------------
    # CARACTERÍSTICAS VISUAIS
    # --------------------------------------------------------

    resultados = {}

    for nome, dados in PLANTAS.items():

        pontos = 0

        cores = dados["cores"]

        # Pontuação pela cor
        if "verde" in cores:
            pontos += verde * 40

        if "amarelo" in cores:
            pontos += amarelo * 40

        if "vermelho" in cores:
            pontos += vermelho * 40

        if "roxo" in cores:
            pontos += roxo * 40

        if "rosa" in cores:
            pontos += rosa * 40

        # Características específicas
        if nome == "Acerola":
            if verde > 0.10:
                pontos += 10
            if vermelho > 0.02:
                pontos += 15

        elif nome == "Ipê-amarelo":
            if amarelo > 0.02:
                pontos += 35

        elif nome == "Ipê-roxo":
            if roxo > 0.01 or rosa > 0.01:
                pontos += 35

        elif nome == "Lajea rubra":
            if vermelho > 0.01 or rosa > 0.01:
                pontos += 30

        elif nome == "Magnólia":
            if rosa > 0.01 or vermelho > 0.01:
                pontos += 20

        elif nome == "Palmeira-imperial":
            # Grande predominância de verde
            if verde > 0.15:
                pontos += 25

        elif nome == "Sibipiruna":
            if verde > 0.15:
                pontos += 20
            if amarelo > 0.01:
                pontos += 20

        resultados[nome] = pontos

    # Ordenar resultados
    resultados = sorted(
        resultados.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return resultados, {
        "verde": verde,
        "amarelo": amarelo,
        "vermelho": vermelho,
        "roxo": roxo,
        "rosa": rosa
    }


# ============================================================
# INTERFACE
# ============================================================

st.title("🌱 PlantAI")
st.subheader("Identificador, classificador e avaliador de plantas")

st.write(
    "Envie uma imagem de uma planta para que o sistema "
    "analise suas características visuais e indique as "
    "espécies mais prováveis."
)

st.divider()

# ============================================================
# UPLOAD
# ============================================================

arquivo = st.file_uploader(
    "📷 Envie uma foto da planta",
    type=["jpg", "jpeg", "png"]
)

if arquivo:

    imagem = Image.open(arquivo).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            imagem,
            caption="Imagem enviada",
            use_container_width=True
        )

    # ========================================================
    # ANÁLISE
    # ========================================================

    resultados, cores = analisar_imagem(imagem)

    melhor_planta = resultados[0][0]
    melhor_pontuacao = resultados[0][1]

    # Converter para porcentagem aproximada
    confianca = min(
        99,
        max(
            1,
            int(melhor_pontuacao)
        )
    )

    with col2:

        st.success("✅ Análise concluída!")

        st.metric(
            "🌿 Planta identificada",
            melhor_planta
        )

        st.metric(
            "📊 Confiança estimada",
            f"{confianca}%"
        )

        dados = PLANTAS[melhor_planta]

        st.write(
            f"**Nome científico:** {dados['nome_cientifico']}"
        )

        st.write(
            f"**Tipo:** {dados['tipo']}"
        )

        st.write(
            f"**Naturalidade:** {dados['naturalidade']}"
        )

    # ========================================================
    # CARACTERÍSTICAS
    # ========================================================

    st.divider()

    st.header("🔎 Características da planta")

    dados = PLANTAS[melhor_planta]

    for caracteristica in dados["caracteristicas"]:
        st.write("•", caracteristica)

    # ========================================================
    # CORES DETECTADAS
    # ========================================================

    st.header("🎨 Cores detectadas na imagem")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "🟢 Verde",
        f"{cores['verde'] * 100:.1f}%"
    )

    col2.metric(
        "🟡 Amarelo",
        f"{cores['amarelo'] * 100:.1f}%"
    )

    col3.metric(
        "🔴 Vermelho",
        f"{cores['vermelho'] * 100:.1f}%"
    )

    col4.metric(
        "🟣 Roxo",
        f"{cores['roxo'] * 100:.1f}%"
    )

    col5.metric(
        "🌸 Rosa",
        f"{cores['rosa'] * 100:.1f}%"
    )

    # ========================================================
    # RANKING
    # ========================================================

    st.divider()

    st.header("🏆 Classificação das espécies")

    for posicao, (nome, pontos) in enumerate(resultados, 1):

        porcentagem = min(100, int(pontos))

        st.write(
            f"**{posicao}º — {nome}**"
        )

        st.progress(
            porcentagem / 100
        )

        st.caption(
            f"Pontuação visual: {porcentagem}%"
        )

    # ========================================================
    # AVISO
    # ========================================================

    st.divider()

    st.info(
        "⚠️ Esta versão utiliza análise visual baseada em cores "
        "e características simples. A porcentagem apresentada "
        "é uma estimativa e não representa uma probabilidade "
        "científica. Para aumentar a precisão, recomenda-se "
        "treinar um modelo de inteligência artificial com "
        "muitas imagens reais de cada espécie."
    )

else:

    st.info(
        "👆 Envie uma foto acima para começar a identificação."
    )

# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "🌱 PlantAI — Sistema experimental de identificação "
    "e classificação de plantas"
)