import argparse
import re
from collections import defaultdict

#predicate_regex = re.compile('([a-zA-Z0-9@=#{\t\s]*?)\((.*?)\)[}]?') #### regex for parsing into 2 groups, but H1,H2 will not be alongside this
predicate_regex = re.compile('([(][a-zA-Z0-9,]+[)])?([a-zA-Z0-9\t\s@=#\{]*[a-zA-Z0-9]+[\t\s]*)\((.*?)\)[}]?') ##### regex for parsing into 3 groups - w/o whitespace removal
#predicate_regex = re.compile('([(][a-zA-Z0-9,]+[)])?([a-zA-Z0-9@=#{\t\s]*?)\((.*?)\)[}]?') ##### regex for parsing into 3 groups - with whitespace removal
#predicate_regex = re.compile('([\(]?[a-zA-Z0-9,]+[\)][a-zA-Z0-9@=#{\t\s]*?)\((.*?)\)[}]?')  #### regex for parsing only (H1,H2) = @entailment

multiple_variable_paranthesis_regex = re.compile('[a-zA-Z0-9,]+')
comment_regex = re.compile("[%]+.*[%]*")
script_split = "#script(python)"
aggregate_predicates = re.compile("(#max|#min)[\s]*{")
equality_constraints_regex = re.compile("[A-Z0-9\"\"]+[!=]+[A-Z0-9\"\"]+")


def create_domain_rule(args, regex_matches, rule):
    print("")
    predicate_matches = (match for match in regex_matches if '@'+args.predicate in match[1])
    variable_stack = []
    visited_predicates = set()
    variable_to_predicate_dict = defaultdict(list)
    predicate_to_variable_dict = defaultdict(list)
    predicate_to_complete_predicate = dict()

    #print("equality_constraints_regex::",equality_constraints_regex.findall(rule))

    rule_head = '{'
    pred_count = 0
    for pred_match in predicate_matches:
        rule_head += 'domain_'+args.replacementPredicate+'('+pred_match[2]+');'
        pred_count += 1
        #print("before strip::",pred_match[2],"split strip:::",pred_match[2].strip().split(','))
        #pred_match[2] = re.sub(r'\s','',pred_match[2])
        variable_stack.extend(re.sub(r'\s','',pred_match[2]).split(','))
        visited_predicates.add('@'+args.predicate)
    rule_head = rule_head.rstrip(';') + '}'
    rule_head = str(pred_count) + rule_head + str(pred_count)
    #print("rule_head::",rule_head)
    rule_body = ""

    for pred in regex_matches:
        
        symbol = ""
        if '=' in pred[1]:
            symbol, pred_name = pred[1].split('=')
            symbol = symbol.strip()
            pred_name = pred_name.strip()
        else:
            pred_name = pred[1].strip()

        pred_name = re.sub(aggregate_predicates,'',pred_name)
        #print("pred::",pred,"pred_name:::",pred_name,"symbol::",symbol)
        #print("symbol::",symbol)

        for var in re.sub(r'\s','',pred[2]).split(','):
            #print("def:::",var, variable_predicate_dict.get(var,[]))
            variable_to_predicate_dict[var].append(pred_name)
            predicate_to_variable_dict[pred_name].extend(pred[2].split(','))
            predicate_to_complete_predicate[pred_name] =  pred[0] + pred[1] + '(' + pred[2] + '),'

        if pred[0]:
            more_variables = multiple_variable_paranthesis_regex.findall(pred[0])[0].split(',')
            for var in more_variables:
                variable_to_predicate_dict[var].append(pred_name)
                predicate_to_variable_dict[pred_name].extend(var)

        if symbol:
            variable_to_predicate_dict[symbol].append(pred_name)
            predicate_to_variable_dict[pred_name].extend(symbol)            


    variable_stack = list(dict.fromkeys(variable_stack))

    while variable_stack:
        current_variable = variable_stack.pop()
        predicates = variable_to_predicate_dict[current_variable]
        
        unvisited_predicates = list(filter(lambda pred: pred not in visited_predicates, predicates))
        visited_predicates.update(unvisited_predicates)
        for pred in unvisited_predicates:
            rule_body += predicate_to_complete_predicate[pred]
            variable_stack.extend(predicate_to_variable_dict[pred])

    equality_constraints = equality_constraints_regex.findall(rule)
    
    if equality_constraints:
        rule_body += ",".join(equality_constraints) + ","

    if args.keepQType:
        question_type_regex = re.compile(args.qTypePredicate+'[(][a-zA-Z0-9,]+[)]')
        qtype_matches = question_type_regex.findall(rule)
        if qtype_matches:
            rule_body += ",".join(qtype_matches)

    final_rule = rule_head + " :- " + rule_body.rstrip(',') + "."
    print("final_rule:::",final_rule)
    return final_rule


def process(args):
    with open(args.dir+args.fname,'r',encoding='utf-8') as theory_file, open(args.dir+'modified1_'+args.fname,'w+',encoding='utf-8') as out_file:
        file_content = theory_file.read()
        parts_of_file = file_content.split(script_split)
        clingo_content = parts_of_file[0]
        clingo_content = re.sub(comment_regex,'',clingo_content)
        rule_file_split = clingo_content.split(".")
        rules_with_predicate = [rule for rule in rule_file_split if '@'+args.predicate in rule]
        rules_without_predicate = [rule for rule in rule_file_split if '@'+args.predicate not in rule]
        for rule in rules_without_predicate:
            if rule.strip():
                out_file.write(rule+".\n")
        out_file.write("%%%%%%changes%%%%%%")
        for i, rule in enumerate(rules_with_predicate):
            
            
            '''Split rule into head and body'''
            rule_head, rule_body = rule.split(':-')
            print("rule::",rule_body)


            '''Find predicates in rule body using the regex. The regex produces a 2 ary tuple of the predicate and the variables.'''
            if '@'+args.predicate in rule_body:
                all_matches = predicate_regex.findall(rule_body)
                predicate_matches = (match for match in all_matches if '@'+args.predicate in match[1])
                new_rule_body = ""
                
                for match in predicate_matches:

                    '''the predicate and the variables joined in string format '''
                    part_of_the_rule_to_replace = (match[1]+'('+match[2]+')').lstrip()
                    symbols,predicate = match[1].split("=")
                    predicate = re.sub(r'\s','',predicate)
                    symbols = re.sub(r'\s','',symbols)

                    replaced_predicate = predicate.replace('@'+args.predicate, args.replacementPredicate)
                    is_aggregate_present = aggregate_predicates.findall(replaced_predicate)
                    if is_aggregate_present:
                        replaced_predicate = replaced_predicate.replace(args.replacementPredicate, '0;'+symbols+","+match[2]+":"+args.replacementPredicate)
                    replaced_predicate = replaced_predicate + "("+match[2] + "," + symbols + ")"
                    
                    if is_aggregate_present:
                        rule = rule.replace(part_of_the_rule_to_replace,symbols + "=" + replaced_predicate)
                    else:
                        rule = rule.replace(part_of_the_rule_to_replace,replaced_predicate)

                domain_rule = create_domain_rule(args, all_matches, rule)

            out_file.write("\n"+rule+".\n")
            out_file.write("\n"+domain_rule+"\n")
        out_file.write("value(0;1).\n")
        out_file.write("\n"+script_split+"\n")
        out_file.write(parts_of_file[1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicate",type=str,default="entailment",help="The function call to be replaced")
    parser.add_argument("--replacementPredicate",type=str,default="validate",help="The predicate used for replacement")
    parser.add_argument("--keepQType",default=True, action="store_false", help="consider question type predicate in the domain definition rule")
    parser.add_argument("--qTypePredicate",type=str,default="qType",help="The name of the question type predicate")
    parser.add_argument("--fname",type=str,default="theory.asp",help="input file")
    parser.add_argument("--dir",type=str,default="./")
    args = parser.parse_args()
    process(args)

if __name__ == "__main__":
    main()