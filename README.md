# FunLLM
LLM最佳学习路径及实践

1. [图解LLM底层技术原理](https://github.com/jason-wang1/FunLLM/blob/master/principles)
   1. [语言模型简介](https://github.com/jason-wang1/FunLLM/blob/main/principles/%E5%9B%BE%E8%A7%A3LLM%E5%BA%95%E5%B1%82%E6%8A%80%E6%9C%AF%E5%8E%9F%E7%90%86/1-%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%AE%80%E4%BB%8B.md)
   2. [图解Attention](https://github.com/jason-wang1/FunLLM/blob/main/principles/%E5%9B%BE%E8%A7%A3LLM%E5%BA%95%E5%B1%82%E6%8A%80%E6%9C%AF%E5%8E%9F%E7%90%86/2-%E5%9B%BE%E8%A7%A3attention.md)
   3. [图解Transformer](https://github.com/jason-wang1/FunLLM/blob/main/principles/%E5%9B%BE%E8%A7%A3LLM%E5%BA%95%E5%B1%82%E6%8A%80%E6%9C%AF%E5%8E%9F%E7%90%86/3-%E5%9B%BE%E8%A7%A3transformer.md)  [Tensorflow keras代码实现Transformer](https://github.com/jason-wang1/FunLLM/blob/main/principles/layers/transformer.py)
   4. [图解BERT](https://github.com/jason-wang1/FunLLM/blob/main/principles/%E5%9B%BE%E8%A7%A3LLM%E5%BA%95%E5%B1%82%E6%8A%80%E6%9C%AF%E5%8E%9F%E7%90%86/4-%E5%9B%BE%E8%A7%A3BERT.md)
   5. [图解GPT](https://github.com/jason-wang1/FunLLM/blob/main/principles/%E5%9B%BE%E8%A7%A3LLM%E5%BA%95%E5%B1%82%E6%8A%80%E6%9C%AF%E5%8E%9F%E7%90%86/5-%E5%9B%BE%E8%A7%A3GPT.md)
2. [应用一：prompt提示词工程](https://github.com/jason-wang1/FunLLM/tree/main/prompts)
   1. [公文笔杆子](https://github.com/jason-wang1/FunLLM/blob/main/prompts/%E5%85%AC%E6%96%87%E7%AC%94%E6%9D%86%E5%AD%90.md)
   2. [岗位职责生成器](https://github.com/jason-wang1/FunLLM/blob/main/prompts/%E5%B2%97%E4%BD%8D%E8%81%8C%E8%B4%A3%E7%94%9F%E6%88%90%E5%99%A8.md)
   3. [爆款文案生成器](https://github.com/jason-wang1/FunLLM/blob/main/prompts/%E7%88%86%E6%AC%BE%E6%96%87%E6%A1%88%E7%94%9F%E6%88%90%E5%99%A8.md)
3. [应用二：对开源LLM进行微调](https://github.com/jason-wang1/FunLLM/tree/main/finetuning)
   1. [下载开源LLM](https://github.com/jason-wang1/FunLLM/blob/main/finetuning/1_download.py)
   2. [使用开源LLM进行推理对话](https://github.com/jason-wang1/FunLLM/blob/main/finetuning/2_chatBot.py)
   3. [使用特定领域样本微调LLM](https://github.com/jason-wang1/FunLLM/blob/main/finetuning/3_loraFineTuning.py)
   4. [测试微调后的效果](https://github.com/jason-wang1/FunLLM/blob/main/finetuning/4_noMergedTest.py)
   5. [合并微调后的文件与原有LLM，生成微调后的模型](https://github.com/jason-wang1/FunLLM/blob/main/finetuning/5_merged.py)
   6. [使用微调后LLM进行推理对话](https://github.com/jason-wang1/FunLLM/blob/main/finetuning/6_chatBotLora.py)
4. 应用三：从零手搓LLM
