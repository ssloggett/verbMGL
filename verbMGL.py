####################################################
# Module 1: generate idiosyncratic rules from corpus
####################################################
# Generate the first set of rules from the data:
# Find the first instance of 'VBZ' or 'VBP' in a sentence, replace it with 'VB____'
# Add a rule to the list of the form [Structural Change, Context]
def generate_idiosyncratic(training, morphology = ['VBZ', 'VBP']):
    """
    Generate the first set of rules from the data. 
    Find the first instance of 'VBZ' or 'VBP' in a sentence, and replace it with 'VB____'.
    Rules are represented as tuple-lists of the form [Structural Change, Context]
    """
    idiosyncratic = {}
    for morpheme in morphology: idiosyncratic[morpheme] = []
    for morpheme in morphology:
        for sentence in training:
            sentence = [[word.split('/')[0],word.split('/')[1]] for word in sentence.split()]
            for i in sentence:
                if i[1] == morpheme:
                    idiosyncratic[i[1]].append(sentence)
                    i[1] = 'VB____'
                    break
    return(idiosyncratic)

# Given two contexts, compare their alignments and keep identical elements. 
# Non-identical elements are collapsed into '*'
def compare(c1,c2,similarity):
    from needleman_wunsch import align
    alignment = align(c1,c2,S=similarity)
    if alignment[0][0] == 'NA': return 'NA'
    
    c = []
    for i in range(0,len(alignment[0])):
        if alignment[0][i][0] in ['[',']'] and alignment[0][i][0] != alignment[1][i][0]:
            return 'NA'
        if alignment[0][i][0] == alignment[1][i][0] in ['[',']'] and (alignment[0][i][1] != alignment[1][i][1]): 
            c.append([alignment[0][i][0], 'XP'])
        elif alignment[0][i][1] == alignment[1][i][1]:
            if alignment[0][i][0] == alignment[1][i][0]:
                c.append(alignment[0][i])
            else: c.append(['*', alignment[0][i][1]])
        else: c.append(['*','*'])
    
    context = [c[0]]
    for i in range(1,len(c)):
        if c[i][1] != '*' or c[i-1][1] != '*': context.append(c[i])
    open_nodes, closed_nodes = [], []
    hasGap = False
    for word in context:
        if word[0] == '[': open_nodes.append(word[1])
        if word[0] == ']':
            if len(open_nodes) == 0: return 'NA'
            elif open_nodes[len(open_nodes)-1] == word[1]:
                open_nodes = open_nodes[0:len(open_nodes)-1]
            else: return 'NA'
        if word[1] == 'VB____': hasGap = True
    if not hasGap or len(open_nodes)>1: return 'NA'
    return context

#######################################################
# Module 2: generalize by comparing rules with same LHS
#######################################################
# Main function for iteratively looping over rules to generalize and create new rules
def generalize_idiosyncratic(idiosyncratic, prior_rules, prior_contexts, similarity_matrix):
    for change in idiosyncratic:
        for i in range(0,len(idiosyncratic[change])):
            for context in idiosyncratic[change][i+1:]:
                new_context = compare(idiosyncratic[change][i], context, similarity_matrix)
                if change in prior_contexts.keys():
                    if new_context != 'NA' and new_context not in prior_contexts[change]:
                        prior_contexts[change].append(new_context)
                        posterior = confidence(change, new_context, idiosyncratic, similarity_matrix)
                        prior_rules[change].append((new_context, posterior))
                        if len(prior_rules[change])%100 == 0: print str(len(prior_rules[change]))+' '+change+' rules created'
                elif new_context != 'NA':
                    prior_contexts[change] = [new_context]
                    posterior = confidence(change, new_context, idiosyncratic, similarity_matrix)
                    prior_rules[change] = [(new_context, posterior)]
                    if len(prior_rules[change])%100 == 0: print str(len(prior_rules[change]))+' '+change+' rules created'
    return [prior_rules, prior_contexts]
    
def generalize(data, similarity_matrix, rules = {}, contexts = {}, morphology = ['VBZ', 'VBP'], printRules = True, fileName = 'bayesMGL_rules.txt'):
    # Generate Idiosyncratic Rules
    idiosyncratic = generate_idiosyncratic(data, morphology)
    
    # Do first-level generalization of idiosyncratic rules
    generalized = generalize_idiosyncratic(idiosyncratic, rules, contexts, similarity_matrix)
    rules = generalized[0]
    contexts = generalized[1]
    print 'Idiosyncratic rules generalized.'
    
    for change in rules:
        for i in range(0,len(rules[change])):
            for context in rules[change][i+1:]:
                new_context = compare(rules[change][i][0], context[0], similarity_matrix)
                if new_context is not 'NA' and new_context not in contexts[change]:
                    contexts[change].append(new_context)
                    posterior = confidence(change, new_context, idiosyncratic, similarity_matrix)
                    rules[change].append((new_context, posterior))
                    if len(rules[change])%100 == 0: print str(len(rules[change]))+' '+change+' rules created'
        print change+' rules generalized'
    if printRules: print_rules(rules, [[change, len(idiosyncratic[change])] for change in idiosyncratic], fileName)
    return [rules,contexts]

def confidence(change, context, train, similarity_matrix):
    prior_denom = 0
    for agreement in train: prior_denom += len(train[agreement])
    prior = float(len(train[change]))/prior_denom
    
    scope = 0
    for sentence in train[change]:
        if compare(context,sentence,similarity_matrix) == context: scope += 1
    likelihood = float(scope)/len(train[change])

    return float(prior)*likelihood
    #return prior
    #return float(likelihood)

def accuracy(data, rules, similarity_matrix, morphology = ['VBZ', 'VBP'], printAcc = True, fileName = 'bayesMGL_results.txt', trainSize = 'NA', grammar = 'NA'):
    print 'Checking accuracy'
    test_data = generate_idiosyncratic(data, morphology)
    
    # Initialize a vector to store choice information
    # {sentence: {observed_morph:, morph1:, morph2:, max: }}

    results = {'sentence': [], 'observed':[], 'predicted':[], 'accuracy':[]}
    denoms, accs = {'total':0}, {'total':0}
    for morpheme in rules: 
        results[morpheme] = []
        denoms[morpheme] = len(test_data[morpheme])
        accs[morpheme] = 0
    denoms['total'] = sum(denoms.values())
    
    for change in test_data:
        for context in test_data[change]:
            max = 0
            choice = 'NA'
            results['sentence'].append(context)
            results['observed'].append(change)
            for morpheme in rules:
                match = 0
                for environment in rules[morpheme]:
                    if compare(context,environment[0],similarity_matrix) == environment[0]: match += environment[1]
                results[morpheme].append(match)
                if match > max: 
                    choice = morpheme
                    max = match
            results['predicted'].append(choice)
            if choice == change: 
                results['accuracy'].append(1)
                accs[change] += 1
                accs['total'] += 1
            else: results['accuracy'].append(0)
    
    for key in denoms: accs[key] = float(accs[key])/denoms[key]
    
    if printAcc: print_accuracy(results, fileName, trainSize, grammar)
    return accs

#######################
# Convenience Functions
#######################
def print_rules(rules, training, fileName):
    import os.path
    print 'Writing to rule file'
    if not os.path.exists(fileName): rules_file = open(fileName, 'w')
    else: 
        rules_file = open(fileName, 'a')
        rules_file.write('\n')
    rules_file.write('#########################\n')
    rules_file.write('Training sentences:\n'+'\t\t'.join([i[0]+': '+str(i[1]) for i in training]))
    rules_file.write('\nTotal rules:\n'+'\t\t'.join([change+': '+str(len(rules[change])) for change in rules]))
    rules_file.write('\n#########################\n')
    rules_file.write('confidence:\trule:\n')
    for change in rules:
            for context in rules[change]:
                rules_file.write(str(format(context[1],'.3f'))+'\t\t0-->'+change+'/ '+' '.join(['/'.join(i) for i in context[0]])+'\n')
    rules_file.close()

def print_accuracy(results, fileName, train, grammar):
    import os.path
    print 'Writing to accuracy file'

    if not os.path.exists(fileName): 
        accuracy_file = open(fileName, 'w')
        accuracy_file.write('grammar\ttrainSize\tobserved\tpredicted\taccuracy\tVBZ\tVBP\n')
    else: accuracy_file = open(fileName, 'a')
    for i in range(0,len(results['observed'])):
        line = [grammar,train, results['observed'][i], results['predicted'][i], results['accuracy'][i], results['VBZ'][i], results['VBP'][i]]
        accuracy_file.write('\t'.join([str(x) for x in line])+'\n')
    accuracy_file.close()
