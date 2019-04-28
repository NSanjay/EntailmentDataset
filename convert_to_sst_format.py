import argparse
import json

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--fname",
						default="stage_difference_dataset.jsonl",
						type=str,
						help="The input file name")
	parser.add_argument("--data_dir",
                        default='./',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	args = parser.parse_args()
	with open(args.data_dir+'filtered_'+args.fname,'r',encoding='utf-8') as input_file, open(args.data_dir+'sst_'+args.fname,'w+',encoding='utf-8') as out_file:
		for a_line in input_file:
			out_line = dict()
			line = json.loads(a_line)
			out_line["sentence1"] = line["premise"]
			out_line["premise"] = line["premise"]
			out_line["sentence2"] = line["hypothesis"]
			out_line["hypothesis"] = line["hypothesis"]
			if line["label"] == "1":
				out_line["gold_label"] = "entailment"
			elif line["label"] == "0":
				out_line["gold_label"] = "neutral"
			out_file.write(json.dumps(out_line)+"\n")


if __name__ == '__main__':
	main()
