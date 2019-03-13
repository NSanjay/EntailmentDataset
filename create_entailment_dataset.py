import argparse
import csv
import os
import re
import json
import time

animal_dict = {'snake':'oviaporous snake'}

filepath = os.path.dirname(os.path.realpath(__file__))
path_to_kb = os.path.join(filepath, 'kb.asp')
path_to_theory = os.path.join(filepath, 'modified_theory.asp')
path_to_query = os.path.join(filepath, 'my_query.asp')
print(path_to_query)
#path_to_seq = os.path.join(filepath, 'seq_ds.asp')

#cmd = "clingo --opt-mode=OptN --verbose=0 --warn no-atom-undefined '"+ path_to_kb + "'  '" + path_to_theory + "' '" + path_to_query +"'"
cmd = "clingo --verbose=0 --warn no-atom-undefined '"+ path_to_kb + "'  '" + path_to_theory + "' '" + path_to_query +"' 20000"
print(cmd)

validate_predicate_reg_ex = re.compile('validate\(\"(.*?)\",\"(.*?)\",(\d)\)')
validate_text_hypothesis_reg_ex = re.compile('\".+\"')
validate_entailment_value_reg_ex = re.compile('\d\)$')


def q_mapping(q_type):
    return {
        "lookup" : "qLookup",
        "difference" : "qStageDifference",
        "indicator" : "qStageIndicator"
    }.get(q_type,"qLookup")
def read_file(parse_dict):

    question_type = parse_dict['qType']
    input_file_directory = './Final Dataset/annotated/'

    input_file_path = input_file_directory + 'type1questions.csv' if not question_type else input_file_directory + 'stage_' + question_type + '.csv'

    output_file_path = 'type1questions_dataset.jsonl' if not question_type else 'stage_' + question_type + '_dataset.jsonl'
    csv_file = open(input_file_path,'r',encoding='utf-8')
    csv_reader = csv.reader(csv_file)

    with open(output_file_path,'w',encoding='utf-8') as out_file:
        #out_file.write("premise\thypothesis\tlabel\n")
        #a_line = list(csv_reader)[54]

        for i, a_line in enumerate(csv_reader):

            organism = re.sub("\d+","",a_line[1].split(" - ")[1])

            #organism = organism if organism not in animal_dict else animal_dict[organism]

            url = a_line[2]
            useOnly = "useOnly(\"" + url + "\")."
            ques = a_line[3]
            ques = ques.replace("\"","\\\"")
            ques = ques.replace("\n","")

            question = "question(\"" + ques + "\")."

            options = a_line[6]


            #replace quotations
            quotes_in_options = re.findall("\"\"(.*?)\"", options)


            for string in quotes_in_options:
                options = re.sub('""'+string+'"',r'"\"'+string+r'\"',options)


            #print(options.split("."))
            #type_predicate,option_a,option_b,_ = options.split(".")
            #type_predicate = type_predicate+"."
            #option_a = option_a+"."
            #option_b = option_b+"."

            #option_a = a_line[4]
            #option_a = "option(\"a\",\"" + option_a + "\")."

            #option_b = a_line[5]
            #option_b = "option(\"b\",\"" + option_b + "\")."

            ans_a = ":- not ans(a)."
            ans_b = ":- ans(b)."

            q_type = "qType(" + q_mapping(question_type) + ")."


            query_file = open(path_to_query, 'w')

            query_file.write(useOnly+"\n"+q_type+"\n"+"\n"+question+"\n"+options+"\n"+ans_a+"\n"+ans_b+"\n#show.")
            query_file.close()

            #print("once")
            #print(cmd)

            output = os.popen(cmd).read()

            #print(output)

            list_of_validate_predicates = set(validate_predicate_reg_ex.findall(output))
            #print(organism,"i::",i+1,"length::",len(list_of_validate_predicates))
            print(organism,"length::",len(list_of_validate_predicates))
            #print("preds::",list_of_validate_predicates)
            for validate_predicate in list_of_validate_predicates:
                #print("i::",i+1,"len",len(validate_predicate))
                to_write_dict = dict()
                to_write_dict["premise"] = validate_predicate[0]
                to_write_dict["hypothesis"] = validate_predicate[1]
                to_write_dict["label"] = validate_predicate[2]
                out_file.write(json.dumps(to_write_dict)+"\n")
                #out_file.write(validate_predicate[0]+"\t"+validate_predicate[1]+"\t"+validate_predicate[2]+"\n")
            time.sleep(10)

def main():
    parser = argparse.ArgumentParser('Create Entailment Dataset')

    parser.add_argument('--qType',default="indicator",choices=["lookup","difference","indicator"])
    parser_dict = vars(parser.parse_args())
    read_file(parser_dict)

if __name__ == '__main__':
    main()