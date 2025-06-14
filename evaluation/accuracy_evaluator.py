import re
from typing import List, Dict, Any
from .base_evaluator import BaseEvaluator

class AccuracyEvaluator(BaseEvaluator):
    """
    用于分类和问答任务的准确率评测器。
    它会尝试从模型输出中提取最终答案。
    """

    def _extract_answer(self, prediction: str) -> str:
        """
        从模型的输出中提取最终答案。
        例如，从 "The answer is X" 中提取 "X"。
        这是一个简化的实现，可以根据需要进行扩展。
        """
        # 尝试匹配 "The answer is X" 类型的模式
        match = re.search(r"(?:the|my)\s+(?:answer|final answer)\s*(?:is|is:)\s*([^\.\n]+)", prediction, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 如果没有特定模式，则返回最后一行非空内容作为答案
        lines = [line.strip() for line in prediction.strip().split('\n') if line.strip()]
        if lines:
            return lines[-1]
        
        return prediction.strip()

    def evaluate(self, predictions: List[str], references: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算预测的准确率。
        对于 GSM8K, 'answer' 字段是答案。
        对于 XCOPA, 'label' 字段是答案的索引 (0或1)，需要和 'choices' 结合。
        """
        correct = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            extracted_pred = self._extract_answer(pred)
            
            if 'answer' in ref: # 适用于 GSM8K
                # 答案通常是数字，需要清洗和标准化
                gold_answer = str(ref['answer']).split('####')[-1].strip()
                extracted_pred_cleaned = re.sub(r"[^0-9\.\-]", "", extracted_pred)
                if extracted_pred_cleaned == gold_answer:
                    correct += 1
            elif 'label' in ref and 'choices' in ref: # 适用于 XCOPA
                gold_label = str(ref['label'])
                if extracted_pred == gold_label or extracted_pred.lower() in ref['choices'][int(gold_label)].lower():
                    correct += 1
            elif 'target' in ref: # 适用于 BBH
                gold_target = str(ref['target'])
                if extracted_pred == gold_target:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy} 