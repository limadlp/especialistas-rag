"""Public Portuguese-language benchmark for the fictional Projeto Cedro."""
from dataclasses import dataclass

# Evidence addresses use book ID and 1-based EPUB section.
DOCUMENTS = {
    "papel": ("Arquivo de papel", (
        ("Umidade", "A faixa de umidade relativa da sala de guarda de papel vai de 42% a 48%. Os sensores registram uma medição a cada quinze minutos para a equipe de conservação."),
        ("Fechamento", "A sala de guarda de papel fica fechada às segundas-feiras para estabilização ambiental. Os pedidos recebidos durante o fechamento entram na fila normal."),
        ("Quarentena", "Caixas de papel recém-recebidas permanecem em quarentena por 36 horas. O prazo começa no recebimento, antes da conferência visual do material."),
    )),
    "fotos": ("Arquivo fotográfico", (
        ("Umidade", "A câmara fotográfica mantém a umidade relativa entre 30% e 36%. A equipe verifica o sensor a cada dez minutos e registra qualquer condensação."),
        ("Fechamento", "A câmara fotográfica fica fechada às quartas-feiras para manutenção. Consultas urgentes são encaminhadas à equipe de conservação para atendimento posterior."),
        ("Quarentena", "Negativos recém-recebidos ficam 60 horas na área de aclimatação. Ao final, a equipe revisa as embalagens antes de encaminhar os filmes à câmara fotográfica."),
    )),
    "remessas": ("Transporte do acervo", (
        ("Veículo", "As caixas de papel são transportadas no veículo Falcão Cinza. O motorista confere o lacre na origem e na chegada e registra os números no recibo."),
        ("Retenção", "O recibo de entrega das caixas de papel é retido por vinte dias no arquivo de expedição. O prazo começa quando o recebimento é confirmado."),
        ("Ocorrências", "Uma caixa de papel com lacre rompido é registrada no formulário Cedro-17. A caixa fica isolada para conferência mesmo quando a etiqueta está legível."),
    )),
    "equipamentos": ("Transporte de equipamentos", (
        ("Veículo", "Os equipamentos de manutenção são transportados no veículo Lince Azul. O entregador confere os números de série na chegada e registra a remessa no manifesto."),
        ("Retenção", "O manifesto dos equipamentos é guardado por quarenta dias, contados da liberação da carga. O recibo de entrega está vinculado ao código da remessa."),
        ("Ocorrências", "Uma carcaça avariada no transporte de equipamentos é registrada no formulário Cedro-24. O técnico fotografa o lado danificado antes de transferir o item."),
    )),
    "vigente": ("Viveiro · norma vigente", (
        ("Irrigação", "Desde março, a norma em vigor determina irrigação por gotejamento às seis da manhã para mudas recém-plantadas. O início do ciclo entra no diário do turno."),
        ("Suspensão", "Segundo a norma em vigor, a irrigação é suspensa quando há pelo menos 12 mm de chuva desde a última leitura. O operador registra o total acumulado."),
        ("Triagem", "Na rotina em vigor, bandejas com mudas ressecadas seguem para o setor de propagação. O responsável anota as condições das raízes no encaminhamento."),
    )),
    "piloto": ("Viveiro · norma descontinuada", (
        ("Irrigação", "Durante o piloto encerrado em fevereiro, as mudas recém-plantadas eram irrigadas às oito da manhã. O horário foi arquivado com a norma descontinuada."),
        ("Suspensão", "Durante o piloto, a irrigação era suspensa com pelo menos 8 mm de chuva desde a última leitura. As medições continuam disponíveis para consulta histórica."),
        ("Triagem", "A área de observação recebia as bandejas com mudas ressecadas durante o piloto. Esse encaminhamento operacional foi encerrado com a revisão da norma."),
    )),
    "revisao": ("Revisão semanal", (
        ("Assinaturas", "O registro de revisão semanal exige duas assinaturas, uma de quem prepara as alterações e outra de quem confere o inventário. O registro conserva as datas."),
        ("Divergências", "A coordenação discute divergências do inventário na reunião de quarta-feira. A equipe conserva o registro original até que a decisão seja cadastrada."),
        ("Correções", "Uma correção aprovada na revisão semanal deve ser lançada em três dias úteis a partir da aprovação. O sistema guarda o identificador da versão anterior."),
    )),
    "auditoria": ("Auditoria anual", (
        ("Assinaturas", "O relatório de auditoria anual exige três assinaturas: a da coordenação, a do auditor e a do responsável pelo acervo. Elas pertencem à versão final."),
        ("Contestações", "As contestações da auditoria anual são avaliadas na reunião de sexta-feira. O auditor disponibiliza o relatório e a contestação à coordenação."),
        ("Correções", "Uma correção aprovada na auditoria anual deve ser lançada em dez dias úteis a partir da comunicação formal. O relatório guarda a data de abertura."),
    )),
}


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    question: str
    evidence: tuple[str, ...] = ()
    answers: tuple[str, ...] = ()  # Each item is required; | separates accepted spellings.


CASES = (
    Case("d01", "direct", "Qual é a faixa de umidade relativa da sala de papel?", ("papel:1",), ("42", "48")),
    Case("d02", "direct", "Em que dia a câmara fotográfica fecha para manutenção?", ("fotos:2",), ("quarta|quartas",)),
    Case("d03", "direct", "Qual é o nome do veículo que transporta caixas de papel?", ("remessas:1",), ("Falcão Cinza",)),
    Case("d04", "direct", "Qual formulário é usado quando chega um equipamento com carcaça avariada?", ("equipamentos:3",), ("Cedro-24",)),
    Case("d05", "direct", "Qual o volume mínimo de chuva para suspender a irrigação segundo a norma em vigor?", ("vigente:2",), ("12 mm|12mm",)),
    Case("d06", "direct", "Quantas assinaturas exige o relatório de auditoria anual?", ("auditoria:1",), ("três|3",)),
    Case("p01", "paraphrase", "Por quanto tempo um lote de papel recém-chegado aguarda antes de entrar na sala de guarda?", ("papel:3",), ("36 horas|trinta e seis horas",)),
    Case("p02", "paraphrase", "Quanto dura a adaptação dos filmes que acabaram de chegar, antes de liberá-los para a câmara?", ("fotos:3",), ("60 horas|sessenta horas",)),
    Case("p03", "paraphrase", "Até quando o comprovante de uma entrega de papel deve continuar arquivado?", ("remessas:2",), ("20 dias|vinte dias",)),
    Case("p04", "paraphrase", "Quem recebe as bandejas de plantas que secaram na rotina atual?", ("vigente:3",), ("propagação",)),
    Case("p05", "paraphrase", "Em qual encontro a coordenação resolve discrepâncias na contagem dos itens?", ("revisao:2",), ("quarta|quartas",)),
    Case("p06", "paraphrase", "Depois da autorização no processo anual, quanto tempo há para registrar uma retificação?", ("auditoria:3",), ("10 dias úteis|dez dias úteis",)),
    Case("c01", "confusable", "Qual sala fecha às segundas, a de papel ou a de fotografia?", ("papel:2",), ("papel",)),
    Case("c02", "confusable", "Quais são os limites de umidade da câmara fotográfica, em vez dos limites do papel?", ("fotos:1",), ("30", "36")),
    Case("c03", "confusable", "Qual formulário se aplica a lacre rompido de caixa de papel, mesmo com etiqueta intacta?", ("remessas:3",), ("Cedro-17",)),
    Case("c04", "confusable", "Por quanto tempo é retido o manifesto de equipamentos, e não o recibo de caixas de papel?", ("equipamentos:2",), ("40 dias|quarenta dias",)),
    Case("c05", "confusable", "Que horário de irrigação deixou de valer em fevereiro?", ("piloto:1",), ("oito da manhã|08:00|8h",)),
    Case("c06", "confusable", "Qual é o prazo da revisão semanal, em vez do prazo da auditoria anual?", ("revisao:3",), ("três dias úteis|3 dias úteis",)),
    Case("m01", "multi_evidence", "Quais faixas de umidade usar para papel e fotografias, respectivamente?", ("papel:1", "fotos:1"), ("42", "48", "30", "36")),
    Case("m02", "multi_evidence", "Quais veículos transportam remessas de papel e de equipamentos de manutenção?", ("remessas:1", "equipamentos:1"), ("Falcão Cinza", "Lince Azul")),
    Case("m03", "multi_evidence", "Como mudou o horário de irrigação do piloto de fevereiro para a norma em vigor desde março?", ("piloto:1", "vigente:1"), ("oito da manhã|08:00|8h", "seis da manhã|06:00|6h")),
    Case("m04", "multi_evidence", "Compare a chuva mínima que suspendia a irrigação no piloto e a que suspende na norma em vigor.", ("piloto:2", "vigente:2"), ("8 mm|8mm", "12 mm|12mm")),
    Case("m05", "multi_evidence", "Quantas assinaturas são exigidas na revisão semanal e na auditoria anual?", ("revisao:1", "auditoria:1"), ("duas|2", "três|3")),
    Case("m06", "multi_evidence", "Em quais dias ocorrem o fechamento da sala de papel e a reunião para contestações da auditoria anual?", ("papel:2", "auditoria:2"), ("segunda|segundas", "sexta|sextas")),
    Case("n01", "unanswerable", "Qual é a temperatura máxima permitida na câmara fotográfica?"),
    Case("n02", "unanswerable", "Qual é a placa do Falcão Cinza?"),
    Case("n03", "unanswerable", "Quantos litros por bandeja são usados na irrigação em vigor?"),
    Case("n04", "unanswerable", "Quem assinou o último relatório anual de auditoria, pelo nome?"),
    Case("n05", "unanswerable", "Qual área tem autorização para substituir um lacre rompido?"),
    Case("n06", "unanswerable", "Qual é o prazo para contestar uma decisão de auditoria anual?"),
)
