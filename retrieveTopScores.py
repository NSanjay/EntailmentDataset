import csv
import json
import re

#scores_file = open('stage_lookup_dataset.jsonl-score.tsv')
stage_name_pattern = re.compile('\:\:(.*?)\:\:')
with open('stage_lookup_dataset.jsonl-score.tsv','r',encoding='utf-8') as scores_file, open('stage_lookup_dataset.jsonl','r',encoding='utf-8') as data_file, open('filtered_scores.jsonl','w',encoding='utf-8') as filtered_file:
	tsv_lines = csv.reader(scores_file,delimiter='\t')
	#dataset_files = json.read()
	i = 0
	prev_number_of_sentences = 0
	prev_row = None
	scores_and_sentences_dict = dict()
	#row = next(tsv)
	for row in tsv_lines:
		#number_of_sentences = row[2]

		row[0],row[1],row[2],row[3] = map(int,row[:4])
		
		row[4] = float(row[4])
		#print(row)
		if i != row[1]:
			i += 1
			sorted_scores = sorted(scores_and_sentences_dict,reverse=True,key=scores_and_sentences_dict.get)
			if prev_row == 1:
			
				highest_sentence = sorted_scores[0]
				if i == 1:
					print("i::",i,highest_sentence,sorted_scores)
			
			else:
			
				index = prev_number_of_sentences // 2
				#print("index:::",index," prev_number_of_sentences::",prev_number_of_sentences)
				highest_sentence = sorted_scores[index]
			

			dataset_line = next(data_file)
			line_in_json = json.loads(dataset_line)
			premise = line_in_json["premise"]
			modified_premise = re.sub(stage_name_pattern,"",premise)
			sentences = modified_premise.split('.')
			best_premise_text = sentences[highest_sentence]
			line_in_json["premise"] = best_premise_text
			filtered_file.write(json.dumps(line_in_json)+"\n")

			scores_and_sentences_dict = dict()


		scores_and_sentences_dict[row[0]] = row[4]
		prev_number_of_sentences = row[2]
		prev_row = row[3]
	
	sorted_scores = sorted(scores_and_sentences_dict,reverse=True,key=scores_and_sentences_dict.get)
	highest_sentence = sorted_scores[0]
	print("highest_sentence::",highest_sentence)
	dataset_line = next(data_file)
	line_in_json = json.loads(dataset_line)
	premise = line_in_json["premise"]
	#print("premise::",premise)
	modified_premise = re.sub(stage_name_pattern,"",premise)
	sentences = modified_premise.split('.')
	best_premise_text = sentences[highest_sentence]
	line_in_json["premise"] = best_premise_text
	filtered_file.write(json.dumps(line_in_json)+"\n")