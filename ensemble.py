import json
from collections import defaultdict

def main():
    '''
    outputs/test_dataset에 앙상블할 K개의 nbest_pred 파일을
    nbest_predictions_1.json
    nbest_predictions_2.json
    ...
    로 저장

    python ensemble.py 로 실행

    outputs/test_dataset에 앙상블 결과 ensemble.json 생성
    '''
    K = 3
    nbests = []
    for _ in range(K):
        with open(f"./outputs/test_dataset/nbest_predictions_{_ + 1}.json", "r") as file:
            nbests.append(json.load(file))

    d = defaultdict(list)
    for nbest in nbests:
        for id_, pred_list in nbest.items():
            for pred in pred_list:
                text = pred["text"]
                probability = pred["probability"]
                flag = 0
                for voting_dist in d[id_]:
                    voting_text = voting_dist["text"]
                    if text == voting_text:
                        voting_dist["probability"] += probability
                        flag = 1
                        break
                if not flag:
                    d[id_].append({"text": text, "probability": probability})
    
    ensemble = {}
    for id_ in d.keys():
        ensemble[id_] = sorted(d[id_], key = lambda x:-x['probability'])[0]['text']

    with open("./outputs/test_dataset/ensemble.json", "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(ensemble, indent=4, ensure_ascii=False) + "\n"
            )

if __name__ == "__main__":
    main()
