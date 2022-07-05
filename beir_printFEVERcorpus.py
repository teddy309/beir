## Code by LSS(github: teddy309)
## date: 2022.06.06 ~ Current(on updating)

import os
import jsonlines

'''
TODO: colloquial 데이터 qrel 형태로 만들기
- from_dir(/home/nlplab/hdd4/lss/dataset_ir/colloquial-claims/colloquial_data/)
- out_dir(/home/nlplab/hdd4/lss/beir/datasets/colloquial/)
- corpus/json.jsonl 하나랑, qrels 안에 (train/test.tsv) 형태로
'''



def main():
    from_dir = os.path.join(os.getcwd(),"datasets/fever/") #'/home/nlplab/hdd4/lss/dataset_ir/colloquial-claims/colloquial_data/'
    #out_dir = os.path.join(os.getcwd(),"datasets/colloquial/")

    colloquial_filenames = ['corpus.jsonl','queries.jsonl']
    printSample = False
    printByTitle, keyTitle = True, 'Bermuda_Triangle'

    if printByTitle:
        data_path = os.path.join(from_dir,'corpus.jsonl')
        with jsonlines.open(data_path) as f:
            for i, line in enumerate(f):
                if line['_id']==keyTitle:
                    print(f'corpus - id:{line["_id"]}, text:{line["text"]}')
                    print(line)
                    

    for filename in colloquial_filenames:
        data_path = os.path.join(from_dir,filename)
        id_list = []

        with jsonlines.open(data_path) as f:
            for i, line in enumerate(f):
                id_list.append(line['_id'])
                #print(f'[{i}] {id_list}') #dict{'colloquial_claims':['str'*3],'fever_claim':"str",'fevr_label':'SUPPORTS','evidences':[],'gold_evidence_set':[],'fever_id':int}
                # corpus:['_id', 'title', 'text', 'metadata']
                # queries:['_id', 'text', 'metadata']
                
                if printSample and i<3:
                    print(f'[{i}] _id:{line["_id"]}, text:{line["text"]}, metadata:{line["metadata"]}')
                '''
                # corpus.jsonl
                id_list: len(5416568), set:5416568, sample:1942_Pittsburgh_Steelers_season
                [0] _id:1928_in_association_football, title:1928 in association football, text:The following are the football ( soccer ) events of the year 1928 throughout the world ., metadata:{}
                [1] _id:1986_NBA_Finals, title:1986 NBA Finals, text:The 1986 NBA Finals was the championship round of the 1985 -- 86 NBA season . It pitted the Eastern Conference champion Boston Celtics against the Western Conference champion Houston Rockets , in a rematch of the 1981 Finals ( only Allen Leavell and Robert Reid remained from the Rockets ' 1981 team ) . The Celtics defeated the Rockets four games to two to win their 16th NBA championship . The championship would be the Celtics ' last until the 2008 NBA Finals . Larry Bird was named the Finals MVP .   On another note , this series marked the first time the `` NBA Finals '' branding was officially used , as they dropped the `` NBA World Championship Series '' branding which had been in use since the beginning of the league , though it had been unofficially called the `` NBA Finals '' for years .   Until the 2011 series , this was the last time the NBA Finals had started before June . Since game three , all NBA Finals games have been played in June . Starting with the following year , the NBA Finals would be held exclusively in the month of June . It was also the last NBA Finals series to schedule a game on a Monday until 1999 and also the last NBA Finals game to be played on Memorial Day .   CBS Sports used Dick Stockton and Tom Heinsohn as the play-by-play man and color commentator respectively . Meanwhile , Brent Musburger was the host and Pat O'Brien ( the Rockets ' sideline ) and Lesley Visser ( the Celtics ' sideline ) were the sideline reporters ., metadata:{}
                [2] _id:1901_Villanova_Wildcats_football_team, title:1901 Villanova Wildcats football team, text:The 1901 Villanova Wildcats football team represented the Villanova University during the 1901 college football season . The Wildcats team captain was John J. Egan ., metadata:{}

                # queries.jsonl
                id_list: len(123142), set:123142, sample:188923
                [0] _id:75397, text:Nikolaj Coster-Waldau worked with the Fox Broadcasting Company., metadata:{'verifiable': 'VERIFIABLE', 'label': 'SUPPORTS', 'evidence': [[[92206, 104971, 'Nikolaj_Coster-Waldau', 7], [92206, 104971, 'Fox_Broadcasting_Company', 0]]]}
                [1] _id:150448, text:Roman Atwood is a content creator., metadata:{'verifiable': 'VERIFIABLE', 'label': 'SUPPORTS', 'evidence': [[[174271, 187498, 'Roman_Atwood', 1]], [[174271, 187499, 'Roman_Atwood', 3]]]}
                [2] _id:214861, text:History of art includes architecture, dance, sculpture, music, painting, poetry literature, theatre, narrative, film, photography and graphic arts., metadata:{'verifiable': 'VERIFIABLE', 'label': 'SUPPORTS', 'evidence': [[[255136, 254645, 'History_of_art', 2]]]}
                '''

        print(f'id_list: len({len(id_list)}), set:{len(set(id_list))}, sample:{id_list[10]}') #다 unique. 갯수대로 나옴.(126502, 8532, 8327)

        print(f'')


    print('hello')


if __name__ == "__main__":
    main()

    print('beir_printFEVERcorpus.py Ended')