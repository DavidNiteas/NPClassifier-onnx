import argparse
from NPClassifier_onnx.npc import NPClassifier

def main():
    parser = argparse.ArgumentParser(description='使用 NPClassifier 进行推理')
    parser.add_argument('--model_dir', type=str, help='模型文件夹路径')
    parser.add_argument('--input_file', type=str, help='输入 Excel 文件路径')
    parser.add_argument('--tag_column', type=str, default='inchi', help='Excel 文件中用于推理的列名')
    parser.add_argument('--batch_size', type=int, default=256, help='每批次处理的样本数量')
    parser.add_argument('--top_k', type=int, default=1, help='返回的前 K 个结果')
    parser.add_argument('--output', type=str, default="./npc_results.xlsx", help='返回的前 K 个结果')

    args = parser.parse_args()

    classifier = NPClassifier(args.model_dir, ["class", "pathway", "superclass"])

    results_df = classifier.predict_from_table_file(
        data_path=args.input_file,
        tag_column=args.tag_column,
        batch_size=args.batch_size,
        top_k=args.top_k
    )
    results_df.to_excel(args.output)

if __name__ == '__main__':
    main()