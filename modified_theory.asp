

%%%%%%%%%%my change %%%%%%%%%%%%%%%
value(0;1).
source(SRC):- description(SRC, life_cycle, O, P).
usable(SRC):- source(SRC),  {useOnly(S):source(S)} 0, forOrganism(O), description(SRC, life_cycle, O, _).
usable(SRC):- useOnly(SRC), 1 {useOnly(S):source(S)}.
forOrganism(O):- qLookup(_,O).
forOrganism(O):- qStageDifference(_,O,_,_).
forOrganism(O):- qStageIndicator(_,O,_).
optionNo(X):- option(X,B).


%%%%%%%%%%lookup%%%%%%%%%%%%%
qType(weighted):-qType(qLookup).
confidence(SRC, X,V):- usable(SRC), qType(qLookup), question(Q), option(X,O), H = @hypothesis(Q,O) , validate(P,H,V), description(SRC,_,_,P).
%confidence(SRC, X,V):- usable(SRC), qType(qLookup), question(Q), option(X,O), H = @hypothesis(Q,O) , V = #max{0;1:validate(P,H)}, description(SRC,_,_,P).
confidence(X,V):- qType(qLookup), V = #max {V1: confidence(SRC, X,V1)}, optionNo(X).
ans(X):- qType(weighted), confidence(X,V), V == #max {V1:confidence(X1,V1)}.


%1{validate(P,H,V) : value(V)}1:- description(SRC,_,_,P), usable(SRC), H = @hypothesis(Q,O) , question(Q), option(_,O), qType(qLookup).
domain_validate(P,H) :- description(SRC,_,_,P), usable(SRC), H = @hypothesis(Q,O) , question(Q), option(_,O), qType(qLookup).

% {validate(P,H)} :- description(SRC,_,_,P), usable(SRC), H = @hypothesis(Q,O) , question(Q), option(_,O), qType(qLookup).
% #minimize {@text_length(P),H:validate(P,H)}.

% validate(P,H,V) :- usable(SRC), confidence(SRC, X,V), question(Q), option(X,O), H = @hypothesis(Q,O) , description(SRC,_,_,P).


%%%%%%%% domain_validate(P, H) :- description(SRC,_,_,P), usable(SRC), H = @hypothesis(Q,O) , question(Q), option(_,O), qType(qLookup).


%%%%%%%%%%%%%%%%%%%%% difference %%%%%%%%%%%%%%%%%%%%
qType(weighted):-qType(qStageDifference).
stage1Id(ID):- qType(qStageDifference), stageAt(SRC, _, _, ID, ST1), usable(SRC), qStageDifference(_,_,ST1,_) .
stage2Id(ID):- qType(qStageDifference), stageAt(SRC, _, _, ID, ST2), usable(SRC), qStageDifference(_,_,_,ST2) .

stage1Name(ST):- qStageDifference(_,_,ST,_), stage1Id(ID1), stage2Id(ID2), ID1<ID2.
stage1Name(ST):- qStageDifference(_,_,_,ST), stage1Id(ID1), stage2Id(ID2), ID1>ID2.

stage2Name(ST):- qStageDifference(_,_,ST,_), stage1Id(ID1), stage2Id(ID2), ID1>ID2.
stage2Name(ST):- qStageDifference(_,_,_,ST), stage1Id(ID1), stage2Id(ID2), ID1<ID2.


%confidence(SRC, X,V):- usable(SRC), qType(qStageDifference), question(Q), option(X,O),
%              (H1,H2) = @hypothesisDifference(Q,O, ST2,ST1), stage2Name(ST2), stage1Name(ST1),
%              H1!="", H2!="", V1 = #max {0;1:validate(P,H1)}, V2 = #max {0;1:validate(P,H2)},
%              V= @multiply(V1,V2), description(SRC,_,_,P).

%confidence(SRC, X,V):- usable(SRC), qType(qStageDifference), question(Q), option(X,O),
%              (H1,H2) = @hypothesisDifference(Q,O, ST2,ST1), stage2Name(ST2), stage1Name(ST1),
%              H1=="", H2=="", H = @hypothesis(Q,O), V= #max {0;1:validate(P,H)}, description(SRC,_,_,P).

confidence(SRC, X,V):- usable(SRC), qType(qStageDifference), question(Q), option(X,O),
              (H1,H2) = @hypothesisDifference(Q,O, ST2,ST1), stage2Name(ST2), stage1Name(ST1),
              H1!="", H2!="", validate(P,H1,V1), validate(P,H2,V2),
              V = @multiply(V1,V2), description(SRC,_,_,P).

confidence(SRC, X,V):- usable(SRC), qType(qStageDifference), question(Q), option(X,O),
              (H1,H2) = @hypothesisDifference(Q,O, ST2,ST1), stage2Name(ST2), stage1Name(ST1),
              H1=="", H2=="", H = @hypothesis(Q,O), validate(P,H,V), description(SRC,_,_,P).

confidence(X,V):- qType(qStageDifference), V = #max {V1: confidence(SRC, X,V1)}, optionNo(X).


%{validate(P,H1)} :- description(SRC,_,_,P), usable(SRC), (H1,_) = @hypothesisDifference(Q,O, ST2,ST1) , H1 != "", stage2Name(ST2), stage1Name(ST1), question(Q), option(_,O), qType(qStageDifference).
%{validate(P,H2)} :- description(SRC,_,_,P), usable(SRC), (_,H2) = @hypothesisDifference(Q,O, ST2,ST1) , H2 != "", stage2Name(ST2), stage1Name(ST1), question(Q), option(_,O), qType(qStageDifference).
%{validate(P,H)} :- description(SRC,_,_,P), usable(SRC), (H1,H2) = @hypothesisDifference(Q,O, ST2,ST1), stage2Name(ST2), stage1Name(ST1), H1=="", H2=="", H = @hypothesis(Q,O) , question(Q), option(_,O), qType(qStageDifference).

%1{validate(P,H1,V):value(V)}1 :- description(SRC,_,_,P), usable(SRC), (H1,_) = @hypothesisDifference(Q,O, ST2,ST1) , H1 != "", stage2Name(ST2), stage1Name(ST1), question(Q), option(_,O), qType(qStageDifference).
%1{validate(P,H2,V):value(V)}1 :- description(SRC,_,_,P), usable(SRC), (_,H2) = @hypothesisDifference(Q,O, ST2,ST1) , H2 != "", stage2Name(ST2), stage1Name(ST1), question(Q), option(_,O), qType(qStageDifference).
%1{validate(P,H,V):value(V)}1 :- description(SRC,_,_,P), usable(SRC), (H1,H2) = @hypothesisDifference(Q,O, ST2,ST1), stage2Name(ST2), stage1Name(ST1), H1=="", H2=="", H = @hypothesis(Q,O) , question(Q), option(_,O), qType(qStageDifference).

domain_validate(P,H1) :- description(SRC,_,_,P), usable(SRC), (H1,_) = @hypothesisDifference(Q,O, ST2,ST1) , H1 != "", stage2Name(ST2), stage1Name(ST1), question(Q), option(_,O), qType(qStageDifference).
domain_validate(P,H2) :- description(SRC,_,_,P), usable(SRC), (_,H2) = @hypothesisDifference(Q,O, ST2,ST1) , H2 != "", stage2Name(ST2), stage1Name(ST1), question(Q), option(_,O), qType(qStageDifference).
domain_validate(P,H) :- description(SRC,_,_,P), usable(SRC), (H1,H2) = @hypothesisDifference(Q,O, ST2,ST1), stage2Name(ST2), stage1Name(ST1), H1=="", H2=="", H = @hypothesis(Q,O) , question(Q), option(_,O), qType(qStageDifference).

%:- validate(P,H1,V1), validate(P,H2,V2), V1 != V2, description(SRC,_,_,P), usable(SRC), (H1,H2) = @hypothesisDifference(Q,O, ST2,ST1) , H1 != "", H2 != "", stage2Name(ST2), stage1Name(ST1), question(Q), option(_,O), qType(qStageDifference).

%%%%%%%%%%%%%%%%%%%%% indicator %%%%%%%%%%%%%%%%%%%%
qType(weighted):-qType(qStageIndicator).
stageId(SRC,ID):- qType(qStageIndicator),stageAt(SRC, _, _, ID, _), usable(SRC).
stageIndicatorId(SRC,ID):- stageAt(SRC, _, _, ID, S), usable(SRC),qStageIndicator(_,_, S).
%trueForStage(SRC, life_cycle, O, Id,X,V):- qStageIndicator(life_cycle,O, S), question(Q), option(X,N),
%                                           usable(SRC), stageId(SRC,Id),
%                                           V = #max {0;1,Text,H : validate(Text, H), description(SRC, life_cycle, O, Text),
%                                           stageAt(SRC, life_cycle, O, Id, S1), H = @hypothesisStageIndicator(Q,N, S, S1)}.

trueForStage(SRC, life_cycle, O, Id,X,V):- qStageIndicator(life_cycle,O, S), question(Q), option(X,N),
                                           usable(SRC), stageId(SRC,Id),
                                           validate(Text, H,V), description(SRC, life_cycle, O, Text),
                                           stageAt(SRC, life_cycle, O, Id, S1), H = @hypothesisStageIndicator(Q,N, S, S1).


res(SRC, life_cycle, O, 1, X , @product("1.0",V,1,ID)) :- trueForStage(SRC, life_cycle, O, 1,X,V), stageIndicatorId(SRC,ID).
res(SRC, life_cycle, O, N, X, @product(V1,V2,N,ID)) :- res(SRC, life_cycle, O, N-1, X , V1), trueForStage(SRC, life_cycle, O, N,X,V2),stageIndicatorId(SRC,ID). % iterative multiplication
finalResult(SRC, life_cycle, O, X , V) :- res(SRC, life_cycle, O, N, X , V),  N = #max {P:stageAt(SRC, life_cycle, O, P, S )}. % the final result

confidence(X,V):- V = #max {Val:finalResult(SRC, life_cycle, O, X , Val)},qType(qStageIndicator), optionNo(X).

%domain_validate(P,H) : description(SRC, life_cycle, O, P), stageAt(SRC, life_cycle, O, Id, S1), H = @hypothesisStageIndicator(Q,N, S, S1)} :-
%                    qStageIndicator(life_cycle,O, S), question(Q), option(X,N), usable(SRC), stageId(SRC,Id).


domain_validate(P,H) :- value(V),  description(SRC, life_cycle, O, P), stageAt(SRC, life_cycle, O, Id, S1), H = @hypothesisStageIndicator(Q,N, S, S1), qStageIndicator(life_cycle,O, S), question(Q), option(X,N), usable(SRC), stageId(SRC,Id).
%1{validate(P,H,V) : value(V),  description(SRC, life_cycle, O, P), stageAt(SRC, life_cycle, O, Id, S1), H = @hypothesisStageIndicator(Q,N, S, S1)}1 :-
%                    qStageIndicator(life_cycle,O, S), question(Q), option(X,N), usable(SRC), stageId(SRC,Id).

1{validate(P,H,V):value(V)}1 :- domain_validate(P,H).

%#minimize {@text_length(P),H :validate(P,H,V)}.
%#minimize {@text_length(P),H:validate(P,H,1)}.


%#show question/1.
%#show option/2.
%#show hypo/1.
#show validate/3.
#show ans/1.
%#show domain_validate/2.
%#show confidence/3.
%#show confidence/2.
%#show stageIndicatorId/2.


#script(python)

import warnings
warnings.filterwarnings("ignore")
from entailment_peter import *
from convert_to_entailment import *
import random

def multiply(x,y):
    print("x::",x)
    print("y::",y)
    if x.string is not None and y.string is not None:
        x = x.string
        y = y.string
    else:
        x = str(x)
        y = str(y)
    print("x:::",x)
    res = float(x) * float(y)
    return str(format(res,'.8f'))

def xnor(x,y):
    if x.string is not None and y.string is not None:
        x = x.string
        y = y.string
    else:
        x = str(x)
        y = str(y)
    if x == y:
        res = 1
    else:
        res = 0
    return str(format(res,'.8f'))


def product(x,y, id, id_target):
    res = 0
    if x.string is not None:
        x = x.string
    else:
        x = str(x)
    if y.string is not None:
        y = y.string
    else:
        y = str(y)
    if id == id_target:
        res = float(x) * float(y)

    else:
        res = float(x) * (1-float(y))

    return str(format(res,'.8f'))


def entailment(text, hyp):
    ent = Entailment()
    res = ent.enatailment(text.string,hyp.string)
    return str(format(res,'.8f'))

def hypothesis(question, op):
    return str(create_hypothesis(get_fitb_from_question(question.string), op.string))

def hypothesisStageIndicator(question, op, stageOriginal, stageNew):
    return str(create_hypothesis_stage_indicator(question.string, op.string, stageOriginal.string, stageNew.string))

def hypothesisDifference(question, op, stage1, stage2):
    h1,h2 = create_hypothesis_comparision(question.string, op.string, stage1.string, stage2.string)
    print("difference:::",str(h1),"2nd::::",str(h2))
    return str(h1),str(h2)

def text_length(text):
    string_length = len(text.string)
    return string_length

#end.