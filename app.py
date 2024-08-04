import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import plotly.graph_objects as go
import networkx as nx
import io
import hashlib
import itertools

st.set_page_config(layout="wide")

# Função para gerar o URL do Gravatar a partir do e-mail
def get_gravatar_url(email, size=100):
    email_hash = hashlib.md5(email.strip().lower().encode('utf-8')).hexdigest()
    gravatar_url = f"https://www.gravatar.com/avatar/{email_hash}?s={size}"
    return gravatar_url

# Definir o e-mail e o tamanho da imagem
email = "marcelo@desenvolvedor.net"  # Substitua pelo seu e-mail
size = 200  # Tamanho da imagem

# Obter o URL do Gravatar
gravatar_url = get_gravatar_url(email, size)

# Layout principal com colunas
col1, col2 = st.columns([1, 3])

# Conteúdo da coluna esquerda
with col1:
    st.markdown(
        f"""
        <div style="text-align: right;">
            <img src="{gravatar_url}" alt="Gravatar" style="width: 250px;">
        </div>
        """,
        unsafe_allow_html=True
    )
# Conteúdo da coluna direita
with col2:
    st.title("Análise Associativa de Dados de Compras de Supermercado")
    st.write("## Marcelo Corni Alves")
    st.write("Julho/2024")
    st.write("Disciplina: Mineração de Dados")

def load_and_validate_data(file_content):
    data = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
    columns = data.columns.tolist()

    invalid_columns = [col for col in columns if not set(data[col].unique()).issubset({0, 1})]

    if invalid_columns:
        raise ValueError(f"Foram encontradas colunas com valores inválidos: {invalid_columns}")

    # Removendo colunas onde todos os valores são 0
    data = data.loc[:, (data != 0).any(axis=0)]

    if columns != data.columns.tolist():
        st.write("Colunas após remoção de colunas com valores inválidos:")
        st.write(data.columns.tolist())

    # Converter para tipo booleano
    data = data.astype(bool)
    
    return data, columns

def analyze_data(data):
    item_counts = data.sum().sort_values(ascending=False)
    total_transactions = len(data)
    item_frequencies = (item_counts / total_transactions).sort_values(ascending=False)
    
    return item_counts, item_frequencies

def generate_stacked_bar_chart(item_frequencies):
    fig = px.bar(
        item_frequencies,
        x=item_frequencies.index,
        y=item_frequencies.values,
        labels={'x': 'Items', 'y': 'Frequency'},
        title='Frequência dos Itens no Conjunto de Dados',
    )
    return fig

def generate_top10_mais_vendidos(data):
    item_frequencies = (data.sum() / len(data)) * 100
    top_items = item_frequencies.nlargest(10)
    top_items_str = top_items.apply(lambda x: f'{x:.2f}%')

    fig = px.bar(
        x=top_items.index,
        y=top_items.values,
        labels={'x': 'Produtos', 'y': 'Percentual de Vendas'},
        title='Top 10 Produtos mais vendidos'
    )

    fig.update_traces(text=top_items_str, textposition='outside')
    fig.update_layout(
        yaxis=dict(
            tickformat='0',
            ticksuffix='%'
        )
    )
    return fig

def generate_top10_menos_vendidos(data):
    item_frequencies = (data.sum() / len(data)) * 100
    top_items = item_frequencies.nsmallest(10)
    top_items_str = top_items.apply(lambda x: f'{x:.2f}%')

    fig = px.bar(
        x=top_items.index,
        y=top_items.values,
        labels={'x': 'Produtos', 'y': 'Percentual de Vendas'},
        title='Top 10 Produtos menos vendidos'
    )

    fig.update_traces(text=top_items_str, textposition='outside')
    fig.update_layout(
        yaxis=dict(
            tickformat='.0',
            ticksuffix='%'
        )
    )
    return fig

def run_apriori(data, min_support=0.01, min_confidence=0.1):
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    return rules

def run_fp_growth(data, min_support=0.01, min_confidence=0.1):
    frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    return rules

def run_eclat(data, min_support=0.01, min_combination=1, max_combination=None):
    def get_support(itemset, transactions):
        return np.sum(np.all(transactions[:, itemset], axis=1)) / transactions.shape[0]

    transactions = data.values
    num_items = transactions.shape[1]
    
    if max_combination is None:
        max_combination = num_items

    itemsets = [[i] for i in range(num_items)]
    frequent_itemsets = []

    current_combination = 1
    while itemsets and current_combination <= max_combination:
        next_itemsets = []
        for itemset in itemsets:
            support = get_support(itemset, transactions)
            if support >= min_support:
                frequent_itemsets.append((itemset, support))
                if current_combination < max_combination:
                    for i in range(itemset[-1] + 1, num_items):
                        next_itemsets.append(itemset + [i])
        itemsets = next_itemsets
        current_combination += 1

    frequent_itemsets = [fi for fi in frequent_itemsets if len(fi[0]) >= min_combination]

    frequent_itemsets_named = [(data.columns[itemset].tolist(), support) for itemset, support in frequent_itemsets]

    return pd.DataFrame(frequent_itemsets_named, columns=['Itemset', 'Support'])

def generate_itemsets_bar_chart(frequent_itemsets):
    frequent_itemsets['Itemset'] = frequent_itemsets['Itemset'].apply(lambda x: ', '.join(x))
    fig = px.bar(
        frequent_itemsets,
        x='Itemset',
        y='Support',
        labels={'Itemset': 'Conjunto de Itens', 'Support': 'Suporte'},
        title='Itemsets Frequentes Gerados pelo ECLAT'
    )
    return fig

def generate_heatmap_itemsets(frequent_itemsets, data):
    itemsets = frequent_itemsets['Itemset'].apply(lambda x: tuple(sorted(x.split(', '))))
    unique_items = sorted(set(itertools.chain.from_iterable(itemsets)))
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}

    itemset_matrix = np.zeros((len(frequent_itemsets), len(unique_items)))

    for idx, row in frequent_itemsets.iterrows():
        for item in row['Itemset'].split(', '):
            itemset_matrix[idx, item_to_idx[item]] = row['Support']

    fig = px.imshow(
        itemset_matrix,
        labels=dict(x="Items", y="Itemsets"),
        x=unique_items,
        y=[', '.join(sorted(itemset)) for itemset in itemsets],
        title='Mapa de Calor dos Itemsets Frequentes Gerados pelo ECLAT'
    )
    return fig

def generate_scatter_plot_eclat(frequent_itemsets):
    fig = px.scatter(
        frequent_itemsets,
        x='Itemset',
        y='Support',
        labels={'Itemset': 'Conjunto de Itens', 'Support': 'Suporte'},
        title='Dispersão dos Itemsets Frequentes Gerados pelo ECLAT'
    )
    return fig

def generate_network_graph_eclat(frequent_itemsets):
    G = nx.Graph()

    for idx, row in frequent_itemsets.iterrows():
        items = row['Itemset'].split(', ')
        for pair in itertools.combinations(items, 2):
            if G.has_edge(pair[0], pair[1]):
                G[pair[0]][pair[1]]['weight'] += row['Support']
            else:
                G.add_edge(pair[0], pair[1], weight=row['Support'])

    pos = nx.spring_layout(G, k=2)

    edge_x = []
    edge_y = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Suporte',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Gráfico de Rede dos Itemsets Frequentes Gerados pelo ECLAT',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper"
                        )],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    return fig



def generate_rules_graph(rules):
    if 'support' in rules.columns and 'confidence' in rules.columns and 'lift' in rules.columns:
        fig = px.scatter(rules, x='support', y='confidence', size='lift', hover_data=['antecedents', 'consequents'])
        fig.layout.xaxis.title = 'Suporte'
        fig.layout.yaxis.title = 'Confiança'
    else:
        fig = go.Figure()
        fig.add_annotation(text="Columns 'support', 'confidence', and 'lift' are required for the scatter plot.",
                           xref="paper", yref="paper",
                           showarrow=False,
                           font=dict(size=20))
    return fig

def generate_parallel_coordinates_plot(rules):
    if 'support' in rules.columns and 'confidence' in rules.columns and 'lift' in rules.columns:
        fig = px.parallel_coordinates(
            rules,
            dimensions=['support', 'confidence', 'lift'],
            color='lift',
            color_continuous_scale=px.colors.diverging.Tealrose,
            title='Gráfico de Coordenadas Paralelas das Regras de Associação'
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Columns 'support', 'confidence', and 'lift' are required for the parallel coordinates plot.",
                           xref="paper", yref="paper",
                           showarrow=False,
                           font=dict(size=20))
    return fig

def generate_heatmap(rules):
    if 'support' in rules.columns and 'confidence' in rules.columns and 'lift' in rules.columns:
        fig = px.density_heatmap(
            rules,
            x='support',
            y='confidence',
            z='lift',
            histfunc='avg',
            color_continuous_scale='Viridis',
            title='Mapa de Calor das Regras de Associação'
        )
        fig.update_layout(
            xaxis_title='Suporte',
            yaxis_title='Confiança',
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Columns 'support', 'confidence', and 'lift' are required for the heatmap.",
                           xref="paper", yref="paper",
                           showarrow=False,
                           font=dict(size=20))
    return fig

def generate_network_graph(rules):
    if 'antecedents' in rules.columns and 'consequents' in rules.columns and 'lift' in rules.columns:
        G = nx.DiGraph()

        for i, rule in rules.iterrows():
            for antecedent in rule['antecedents']:
                for consequent in rule['consequents']:
                    G.add_edge(antecedent, consequent, weight=rule['lift'])

        pos = nx.spring_layout(G, k=2)
        
        edge_x = []
        edge_y = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_text = []
        hover_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            hover_text.append(f"{node}<br># de conexões: {len(list(G.neighbors(node)))}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            hovertext=hover_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Conexões',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        node_adjacencies = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))

        node_trace.marker.color = node_adjacencies

        fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                    title='Gráfico de Rede de Regras de Associação',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper"
                    )],
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False))
                    )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Columns 'antecedents', 'consequents', and 'lift' are required for the network graph.",
                           xref="paper", yref="paper",
                           showarrow=False,
                           font=dict(size=20))
    return fig

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Selecione o arquivo CSV:", type="csv")

if uploaded_file is not None:
    content = uploaded_file.read()
    data, columns = load_and_validate_data(content)

    # Texto a ser exibido dentro do quote
    quote_text = f"""
    > Colunas do arquivo CSV de um
    > total de {len(data)} registros
    """
    # Exibir o texto dentro do quote
    st.markdown(quote_text)

    st.write(columns)

    st.write(f"")

    # Analisar e mostrar os dados
    item_counts, item_frequencies = analyze_data(data)
    st.plotly_chart(generate_stacked_bar_chart(item_frequencies))
    st.plotly_chart(generate_top10_mais_vendidos(data))
    st.plotly_chart(generate_top10_menos_vendidos(data))
    
    # Selecionar algoritmo
    algorithm = st.selectbox(
        "Escolha o algoritmo para aplicar regras de associação:",
        ["Apriori", "Eclat", "FP-Growth"]
    )

    min_support = st.slider("Suporte Mínimo", 0.01, 0.5, 0.01)

    if algorithm == "Apriori":
        min_confidence = st.slider("Confiança Mínima", 0.1, 1.0, 0.1)
    elif algorithm == "Eclat":
        min_combination = st.slider("Combinação Mínima", 1, 10, 1)
        max_combination = st.slider("Combinação Máxima", 1, 10, 1)
    elif algorithm == "FP-Growth":
        min_confidence = st.slider("Confiança Mínima", 0.1, 1.0, 0.1)
    

    if st.button("Executar Algoritmo"):
        try:
            if algorithm == "Apriori":
                rules = run_apriori(data, min_support, min_confidence)
                st.dataframe(rules)
                st.plotly_chart(generate_rules_graph(rules))
                st.plotly_chart(generate_parallel_coordinates_plot(rules))
                st.plotly_chart(generate_heatmap(rules))
                st.plotly_chart(generate_network_graph(rules))
            elif algorithm == "Eclat":
                frequent_itemsets = run_eclat(data, min_support, min_combination, max_combination)
                st.dataframe(frequent_itemsets)
                st.plotly_chart(generate_itemsets_bar_chart(frequent_itemsets))
                st.plotly_chart(generate_heatmap_itemsets(frequent_itemsets, data))
                st.plotly_chart(generate_scatter_plot_eclat(frequent_itemsets))
                st.plotly_chart(generate_network_graph_eclat(frequent_itemsets))
            elif algorithm == "FP-Growth":
                rules = run_fp_growth(data, min_support, min_confidence)
                st.dataframe(rules)
                st.plotly_chart(generate_rules_graph(rules))
                st.plotly_chart(generate_parallel_coordinates_plot(rules))
                st.plotly_chart(generate_heatmap(rules))
                st.plotly_chart(generate_network_graph(rules))
        except ValueError as e:
            st.error(e)
    
# Manter o estado dos dados
if "data" not in st.session_state:
    st.session_state.data = None
if "columns" not in st.session_state:
    st.session_state.columns = None

if uploaded_file is not None:
    st.session_state.data, st.session_state.columns = load_and_validate_data(content)

if st.session_state.data is not None:
    data, columns = st.session_state.data, st.session_state.columns
