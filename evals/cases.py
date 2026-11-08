"""Fictional reference passages and independently specified expected answers."""

DOCUMENTS = {
    "conservacao": {
        "title": "Manual de conservação do acervo",
        "text": (
            "No arquivo do Projeto Cedro, o depósito de papel fica fechado às segundas-feiras "
            "para estabilização ambiental. A faixa de umidade relativa recomendada para esse "
            "depósito é de 42 a 48 por cento. O sensor registra uma medição a cada quinze minutos."
        ),
    },
    "transporte": {
        "title": "Protocolo de transporte do acervo",
        "text": (
            "O transporte de caixas do Projeto Cedro usa o veículo identificado como Falcão Cinza. "
            "O motorista deve confirmar o lacre na origem e na chegada. A ficha de transporte "
            "é armazenada por vinte dias no arquivo de expedição, junto ao recibo de entrega."
        ),
    },
    "viveiro": {
        "title": "Rotina do viveiro",
        "text": (
            "No Projeto Cedro, as mudas recém-plantadas recebem água por gotejamento às "
            "seis da manhã. Em dias de chuva, a irrigação é suspensa com base no pluviômetro. "
            "Bandejas com mudas ressecadas seguem para análise no setor de propagação."
        ),
    },
    "revisao": {
        "title": "Política de revisão de registros",
        "text": (
            "No Projeto Cedro, o registro de revisão semanal exige duas assinaturas. Uma é da "
            "pessoa que prepara as alterações e a outra é de quem confere o inventário. "
            "Divergências são encaminhadas para a coordenação na reunião de quarta-feira."
        ),
    },
}

CASES = [
    ("Qual faixa de umidade relativa usar para conservar o papel?", "conservacao", ("42 a 48", "42% a 48%", "42–48")),
    ("Em que dia a sala de guarda de papel fica indisponível?", "conservacao", ("segunda", "segunda-feira", "segundas-feiras")),
    ("Como se chama o veículo que faz o transporte de caixas?", "transporte", ("Falcão Cinza",)),
    ("Por quantos dias se guarda o documento de entrega?", "transporte", ("vinte dias", "20 dias")),
    ("Qual é o horário da irrigação das mudas novas?", "viveiro", ("seis da manhã", "06:00", "6h")),
    ("Quantas pessoas devem assinar a revisão da semana?", "revisao", ("duas", "2 assinaturas")),
]
