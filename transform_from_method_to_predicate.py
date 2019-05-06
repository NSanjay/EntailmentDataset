import argparse
import re

predicate_regex = re.compile('([a-zA-Z0-9@=#{\t\s]*?)\((.*?)\)[}]?')
comment_regex = re.compile("[%]+.*[%]*")
script_split = "#script(python)"
aggregate_predicates = re.compile("(#max|#min){")

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
            out_file.write(rule+"\n")
        out_file.write("%%%%%%changes%%%%%%")
        for i, rule in enumerate(rules_with_predicate):
            
            '''Split rule into head and body'''
            rule_head, rule_body = rule.split(':-')
            print("rule::",rule_body)

            '''Find predicates in rule body using the regex. The regex produces a 2 ary tuple of the predicate and the variables.'''
            all_matches = predicate_regex.findall(rule_body)
            new_rule_body = ""
            for match in all_matches:
                
                if args.predicate in match[0]: # If the predicate to be substituted is in the split predicate
                    
                    '''the predicate and the variables joined in string format '''
                    part_of_the_rule_to_replace = (match[0]+'('+match[1]+')').lstrip()
                    print("match::",part_of_the_rule_to_replace)
                    symbols,predicate = match[0].split("=")
                    predicate = re.sub(r'\s','',predicate)
                    symbols = re.sub(r'\s','',symbols)
                    #print("predicate::", predicate,symbols)

                    replaced_predicate = predicate.replace('@'+args.predicate, args.replacementPredicate)
                    is_aggregate_present = aggregate_predicates.findall(replaced_predicate)
                    if is_aggregate_present:
                        replaced_predicate = replaced_predicate.replace(args.replacementPredicate, '0;'+symbols+","+match[1]+":"+args.replacementPredicate)
                    replaced_predicate = replaced_predicate + "("+match[1] + "," + symbols + ")"
                    
                    print("replaced_var:::",replaced_predicate)
                    if is_aggregate_present:
                        rule = rule.replace(part_of_the_rule_to_replace,symbols + "=" + replaced_predicate)
                    else:
                        rule = rule.replace(part_of_the_rule_to_replace,replaced_predicate)
                    print("after replace::",rule)

            #print("matches",i,"::",all_matches)
            out_file.write(rule+"\n")
        out_file.write("value(0;1).\n")
        out_file.write("\n"+script_split+"\n")
        out_file.write(parts_of_file[1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicate",type=str,default="entailment",help="The function call to be replaced")
    parser.add_argument("--replacementPredicate",type=str,default="validate",help="The predicate used for replacement")
    parser.add_argument("--fname",type=str,default="theory.asp",help="input file")
    parser.add_argument("--dir",type=str,default="./")
    args = parser.parse_args()
    process(args)

if __name__ == "__main__":
	main()