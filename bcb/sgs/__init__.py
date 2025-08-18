# --- 1. IMPORTAÇÕES ---
import json as json_parser
from io import StringIO
from typing import Dict, Generator, List, Optional, Tuple, TypeAlias, Union
import datetime as dt

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

from bcb.utils import Date, DateInput

"""
Sistema Gerenciador de Séries Temporais (SGS)

O módulo ``sgs`` obtem os dados do webservice do Banco Central,
interface json do serviço BCData/SGS -
`Sistema Gerenciador de Séries Temporais (SGS)
<https://www3.bcb.gov.br/sgspub/localizarseries/localizarSeries.do?method=prepararTelaLocalizarSeries>`_.
"""

# --- 2. CLASSES E FUNÇÕES DE BASE ---
class SGSCode:
    """Representa um código de série temporal do SGS com nome e valor."""
    def __init__(self, code: Union[str, int], name: Optional[str] = None) -> None:
        if name is None:
            if isinstance(code, int) or isinstance(code, str):
                self.name = str(code)
                self.value = int(code)
        else:
            self.name = str(name)
            self.value = int(code)

    def __repr__(self):
        return f"{self.value} - {self.name}" if self.name else f"{self.value}"


SGSCodeInput: TypeAlias = Union[
    int, 
    str, 
    Tuple[str, Union[int, str]], 
    List[Union[int, str, Tuple[str, Union[int, str]]]], 
    Dict[str, Union[int, str]],
    ]


def _codes(codes: SGSCodeInput) -> Generator[SGSCode, None, None]:
    """
    Normaliza diferentes formatos de entrada de códigos de séries em um gerador
    de objetos SGSCode.
    """
    if isinstance(codes, int) or isinstance(codes, str):
        yield SGSCode(codes)
    elif isinstance(codes, tuple):
        yield SGSCode(codes[1], codes[0])
    elif isinstance(codes, list):
        for cd in codes:
            _ist = isinstance(cd, tuple)
            yield SGSCode(cd[1], cd[0]) if _ist else SGSCode(cd)
    elif isinstance(codes, dict):
        for name, code in codes.items():
            yield SGSCode(code, name)


def _get_url_and_payload(code: int, start_date: Optional[DateInput], end_date: Optional[DateInput], last: int) -> Dict[str, str]:
    """Monta a URL e os parâmetros para a chamada à API do SGS."""
    payload = {"formato": "json"}
    if last == 0:
        if start_date is not None or end_date is not None:
            payload["dataInicial"] = Date(start_date).date.strftime("%d/%m/%Y")
            end_date = end_date if end_date else "today"
            payload["dataFinal"] = Date(end_date).date.strftime("%d/%m/%Y")
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
    else:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/{last}"
    return {"payload": payload, "url": url}


def _format_df(df: pd.DataFrame, code: SGSCode, freq: str) -> pd.DataFrame:
    """
    Formata o DataFrame bruto retornado pela API, renomeando colunas,
    convertendo tipos de dados e definindo o índice.
    """
    cns = {"data": "Date", "valor": code.name, "datafim": "enddate"}
    df = df.rename(columns=cns)
    if "Date" in df:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    if "enddate" in df:
        df["enddate"] = pd.to_datetime(df["enddate"], dayfirst=True)
    df = df.set_index("Date")
    if freq:
        df.index = df.index.to_period(freq)
    return df


# --- 3. FUNÇÕES AUXILIARES PARA A LÓGICA DE BUSCA AVANÇADA ---

def _probe_is_daily(code: int) -> bool:
    """
    Sonda a API buscando os 2 últimos pontos para determinar se a série é diária.
    A verificação é feita calculando a diferença em dias entre os dois últimos pontos.

    Args:
        code (int): O código da série a ser verificada.

    Returns:
        bool: True se a diferença for de 1 dia (indicando série diária),
              False caso contrário. Em caso de erro, assume True por segurança.
    """
    try:
        # Pede os dois últimos pontos da série para inferir a frequência.
        urd = _get_url_and_payload(code, None, None, last=2)
        res = requests.get(urd["url"], params=urd["payload"])
        res.raise_for_status()
        data = json_parser.loads(res.text)

        # Se houver menos de 2 pontos, não é possível inferir a frequência.
        if len(data) < 2:
            return False

        # Converte as datas (formato "DD/MM/YYYY") e calcula a diferença.
        date_format = "%d/%m/%Y"
        last_date = dt.datetime.strptime(data[1]['data'], date_format).date()
        penultimate_date = dt.datetime.strptime(data[0]['data'], date_format).date()
        delta_days = (last_date - penultimate_date).days
        return delta_days <= 20
    except (requests.RequestException, IndexError, KeyError, ValueError):
        # Se a sondagem falhar, assume o pior cenário (série diária) para
        # garantir que a busca em blocos seja acionada se o período for longo.
        return True


def _fetch_sgs_json(code: int, start: Optional[DateInput], end: Optional[DateInput], last: int) -> str:
    """
    Executa uma única requisição HTTP para a API do SGS e retorna o JSON como texto.

    Args:
        code (int): Código da série.
        start (DateInput, optional): Data de início.
        end (DateInput, optional): Data de fim.
        last (int): Número de últimas observações a serem retornadas.

    Returns:
        str: A resposta da API em formato JSON de texto.
    """
    urd = _get_url_and_payload(code, start, end, last)
    res = requests.get(urd["url"], params=urd["payload"])
    res.raise_for_status()  # Lança um erro para status HTTP de falha (4xx ou 5xx)
    return res.text


def _fetch_in_chunks(code: int, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Busca os dados de uma série em blocos de tempo para contornar limites da API.
    Converte cada bloco em um DataFrame e os concatena no final.

    Args:
        code (int): Código da série.
        start_date (dt.date): Data de início do período total.
        end_date (dt.date): Data de fim do período total.

    Returns:
        pd.DataFrame: Um DataFrame contendo todos os dados do período solicitado.
    """
    df_list = []
    current_start = start_date
    while current_start < end_date:
        # Define o fim do bloco (chunk) atual.
        current_end = current_start + relativedelta(years=5)
        if current_end > end_date:
            current_end = end_date
        
        try:
            json_text = _fetch_sgs_json(code, current_start, current_end, 0)
            if json_text:
                # Converte o JSON do bloco em um DataFrame e o adiciona à lista.
                chunk_df = pd.read_json(StringIO(json_text), orient='records')
                if not chunk_df.empty:
                    df_list.append(chunk_df)
        except (json_parser.JSONDecodeError, requests.exceptions.HTTPError):
            # Ignora blocos que retornam resposta vazia ou com erro, continuando o processo.
            pass
            
        # Avança para o próximo bloco.
        current_start = current_end + relativedelta(days=1)
    
    if not df_list:
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)


def get_json_as_df(code: int, start: Optional[DateInput] = None, end: Optional[DateInput] = None, last: int = 0) -> pd.DataFrame:
    """
    Função central que orquestra a busca de dados, decidindo entre uma busca
    única ou em blocos, e sempre retorna um DataFrame.

    Args:
        code (int): Código da série.
        start (DateInput, optional): Data de início.
        end (DateInput, optional): Data de fim.
        last (int): Número de últimas observações.

    Returns:
        pd.DataFrame: DataFrame com os dados brutos da série ("data", "valor").
    """
    # Caso 1: Busca pelas últimas 'n' observações.
    if last > 0:
        text = _fetch_sgs_json(code, None, None, last)
        return pd.read_json(StringIO(text), orient='records')

    # Caso 2: Datas de início e fim são fornecidas.
    if start:
        start_date = pd.to_datetime(start).date()
        end_date = pd.to_datetime(end or dt.date.today()).date()
        is_long_period = (end_date - start_date).days > 3600
        
        # Lógica proativa: se o período for longo, sonda a frequência.
        if is_long_period and _probe_is_daily(code):
            return _fetch_in_chunks(code, start_date, end_date)
        
        # Se não for longo ou não for diário, faz uma busca única.
        text = _fetch_sgs_json(code, start_date, end_date, 0)
        return pd.read_json(StringIO(text), orient='records')
    
    # Caso 3: Nenhuma data fornecida (busca a série completa).
    else:
        # Lógica reativa: tenta a busca completa. Se falhar, assume que é uma
        # série diária longa e recorre à busca em blocos.
        try:
            text = _fetch_sgs_json(code, None, None, 0)
            return pd.read_json(StringIO(text), orient='records')
        except (requests.exceptions.HTTPError, ValueError):
            # O fallback inicia a busca a partir de uma data antiga e segura.
            start_date = dt.date(1980, 1, 1)
            end_date = dt.date.today()
            return _fetch_in_chunks(code, start_date, end_date)


# --- 4. FUNÇÃO PRINCIPAL (INTERFACE PÚBLICA) ---

def get(
    codes: SGSCodeInput,
    start: Optional[DateInput] = None,
    end: Optional[DateInput] = None,
    last: int = 0,
    multi: bool = True,
    freq: Optional[str] = None,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Retorna um DataFrame pandas com séries temporais obtidas do SGS.
    Esta função contorna a limitação de 10 anos para séries diárias ao
    realizar múltiplas requisições em blocos quando necessário.

    Args:
        codes ({int, str, list, dict}): Código(s) da(s) série(s) a ser(em) consultada(s).
        start (DateInput, optional): Data de início da série.
        end (DateInput, optional): Data de fim da série.
        last (int, optional): Retorna os 'n' últimos dados disponíveis.
        multi (bool, optional): Se True, retorna um único DataFrame para múltiplas
                                séries. Se False, uma lista de DataFrames.
        freq (str, optional): Frequência a ser utilizada no índice do DataFrame.

    Returns:
        Union[pd.DataFrame, List[pd.DataFrame]]: DataFrame (ou lista de DataFrames)
                                                 com as séries temporais.
    """
    dfs = []
    for code in _codes(codes):
        # A função get_json_as_df abstrai toda a complexidade da busca.
        raw_df = get_json_as_df(code.value, start, end, last)
        
        if raw_df.empty:
            continue
        
        # Formata o DataFrame bruto para o padrão final.
        df = _format_df(raw_df, code, freq)
        
        # Garante a remoção de duplicatas no índice (uma camada extra de segurança).
        if not df.index.is_unique:
            df = df.loc[~df.index.duplicated(keep='first')]
        
        # Garante que os dados retornados comecem na data de início solicitada.
        if start:
            start_date_dt = pd.to_datetime(start)
            df = df.loc[df.index >= start_date_dt]
        
        dfs.append(df)
        
    if not dfs:
        return pd.DataFrame()
    
    if len(dfs) == 1:
        return dfs[0]
    else:
        if multi:
            return pd.concat(dfs, axis=1)
        else:
            return dfs