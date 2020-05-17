# perform arithmetic operations. 
def applyOp(a, b, op): 
      
    if op == '+': return a + b 
    if op == '-': return a - b 
    if op == '*': return a * b 
    if op == '/': return a / b 

# find precedence of operators in equation 
def precedence(op): 
      
    if op == '+' or op == '-': 
        return 1
    if op == '*' or op == '/': 
        return 2
    return 0
      
# evaluate the result of equation 
def evaluate(eqn):   
    i = 0
    # stack to store operators
    ops = [] 
    # stack to store integer values 
    values = [] 
    
    while i < len(eqn):  
        # skip if the element is empty space 
        if eqn[i] == ' ': 
            i += 1
            continue 
            
        # element is a number  
        elif eqn[i].isdigit(): 
            val = 0   
            # if more there are more than one digits in the number 
            while (i < len(eqn) and eqn[i].isdigit()): 
                val = (val * 10) + int(eqn[i]) 
                i += 1    
            values.append(val) 
          
        # element is an op
        else: 
            while (len(ops) != 0 and
                precedence(ops[-1]) >= precedence(eqn[i])):            
                val2 = values.pop() 
                val1 = values.pop() 
                op = ops.pop()   
                values.append(applyOp(val1, val2, op))  
            # push current element to 'ops'. 
            ops.append(eqn[i]) 
        i += 1
      
    # apply ops 
    while len(ops) != 0:    
        val2 = values.pop() 
        val1 = values.pop() 
        op = ops.pop()            
        values.append(applyOp(val1, val2, op))     
    # Return result at top of values 
    return values[-1] 