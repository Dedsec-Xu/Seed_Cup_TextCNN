#to attach train_b train_a valid_a
with open('./data/headless/head.txt', 'r') as f1:
    with open('./data/headless/train_b.txt', 'r') as f2:
        with open('./data/headless/train_a.txt', 'r') as f3:
            with open('./data/headless/train_final_2.txt','w') as f5:
                f1=open('./data/headless/head.txt', 'r')
                f2=open('./data/headless/valid_b.txt', 'r')
                f3=open('./data/headless/valid_a.txt', 'r')
                f5=open('./data/headless/valid_final_2.txt','w')
                
                data=f1.read()
                f5.write(data)
                f1.close()
                
                data=f2.read()
                f5.write(data)
                f2.close()
                   
                data=f3.read()
                f5.write(data)
                f3.close()
                
                f5.close()
