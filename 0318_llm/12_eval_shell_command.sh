# 1、先在shell终端执行这个命令，构建judge_modge_args参数，主要需要保证DEEPSEEK_API_KEY有值
export JUDGE_MODEL_ARGS="$(python - <<'PY'
import os, json

args = {
      "model_id": "deepseek-v4-pro",
      "api_url": "https://api.deepseek.com",
      "api_key": os.environ["DEEPSEEK_API_KEY"],
      "generation_config": {
          "temperature": 0.0,
          "max_tokens": 1024
      },
      "score_type": "numeric",
      "score_pattern": "\[\[(\d+(?:\.\d+)?)\]\]",
      "prompt_template": """你是一个中文关键词抽取评测裁判。不要输出分析过程。

  请判断模型抽取的关键词的效果。最终输出一个分数。
  最终只输出一个分数，格式必须为 [[0到1之间的小数]]，例如 [[0.8]]

示例：
任务：
抽取出文本中的关键词：\n标题：人工神经网络在猕猴桃种类识别上的应用\n文本：在猕猴桃介电特性研究的基础上,将人工神经网络技术应用于猕猴桃的种类识别.该种类识别属于模式识别,其关键在于提取样品的特征参数,在获得特征参数的基础上,选取合适的网络通过训练来进行识别.猕猴桃种类识别的研究为自动化识别果品的种类、品种和新鲜等级等提供了一种新方法,为进一步研究果品介电特性与其内在品质的关系提供了一定的理论与实践基础.
模型输出：
食品科学技术基础学科;猕猴桃;应用;人工神经网络;介电特性;识别
你的最终输出：[[0.9]]


评分标准：
1、模型是否按照特定格式输出，权重：0.6
2、所提取的内容，和原文的关联程度，权重：0.4
最终将结果限制到0-1的范围当中

  [题目]
  {question}


  [模型输出]
  {pred}


  输出一个最终的0-1的分数"""
  }

print(json.dumps(args, ensure_ascii=False))
PY
)"

# 2、然后再执行这个参数，完成评估
evalscope eval \
    --model Qwen3-0.6B \
    --api-url "http://127.0.0.1:8000/v1" \
    --datasets general_qa \
    --dataset-args '{"general_qa":{"local_path":"benchmark/custom_eval/text/qa","subset_list":["keyword_extraction"],"prompt_template":"{query}"}}' \
    --judge-strategy llm \
    --judge-model-args "$JUDGE_MODEL_ARGS" \
    --generation-config '{"temperature":0.0,"max_tokens":256}' \
    --eval-batch-size 4


