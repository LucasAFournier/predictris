SIZE = 30
SIZE_NEXT = 25
SIZE_LANE = 20
moveX = 5
moveY = 0
scene = 0

def RMG():
    global tetro_bag
    type_mino = 0
    if tetro_bag == []:
        tetro_bag = [i for i in range(0, tl)]
    type_mino = random.choice(tetro_bag)
    tetro_bag.remove(type_mino)
    return type_mino

type_c = RMG()
type_next = RMG()
type_0 = RMG()
type_1 = RMG()
type_2 = RMG()
type_3 = RMG()
type_4 = RMG()
type_5 = RMG()
    
def make_field(field_list = None):
    fielded = []
    if field_list != None:
        if type(field_list) is list:
            fielded = field_list
    else:
        for y in range(26):
            sub = []
            for x in range(16):
                if x==0 or x>=11 or y>=21 :
                    sub.append(40)
                else :
                    sub.append(50)
            fielded.append(sub)
    return fielded

field = make_field()
field0 = make_field()

nextfield = []
for y in range(20):
    sub = []
    for x in range(20):
        sub.append(50)
    nextfield.append(sub)

lanefield = []
for y in range(120):
    sub = []
    for x in range(20):
        sub.append(50)
    lanefield.append(sub)

def keyPress(event):   
    global moveX, moveY, type_0, type_1, type_2, type_3, type_4, type_5, score, ...
    afterX = moveX
    afterY = moveY
    afterTetro = []
    afterTetro.extend(tetro[type_c][turn])
    parsed = False
    if scene == 0:
        ...
    elif scene == 1:
        if event.keysym==cont["M_Right"] :
            if hardd is False and pause is False:
                afterX += 1
                move_s.play()
                if judge(afterX, afterY - 1, afterTetro) is True and spin is True:
                    spin = False
                    tspin = False
                    tmini = False
        elif event.keysym==cont["M_Left"] :
            if hardd is False and pause is False:
                afterX -= 1
                move_s.play()
                if judge(afterX, afterY - 1, afterTetro) is True and spin is True:
                    spin = False
                    tspin = False
                    tmini = False
        elif event.keysym==cont["SDrop"] :
            if pause is False:
                afterY += 1
                stoke = judge(afterX, afterY, afterTetro) 
                if stoke is True:
                    spin = False
                    soft_s.play()
                    tspin = False
                    tmini = False
                    if glob["scoremode"] == "sega":
                        if level <= 2:
                            score += int(1)
                        elif level <= 4:
                            score += int(2)
                        elif level <= 6:
                            score += int(3)
                        elif level <= 8:
                            score += int(4)
                        else:
                            score += int(5)
                    elif glob["scoremode"] == "bps":
                        score += 0
                    else:
                        score += 1
        elif event.keysym==cont["T_Right"] :
            spin = True
            cout = []
            turn_s.play()
            if hardd is False and pause is False:
                parse = True
                if type_c in even_pic:
                    kick = wallkickC[turn+4]
                else:
                    kick = wallkickC[turn]
                turn += 1
                if turn > 3:
                    turn -= 4
                cout.extend(afterTetro)
                afterTetro.clear()
                afterTetro.extend(tetro[type_c][turn])
                if type_c not in sym_pic:
                    for (offX, offY) in kick:
                        parse = judge(afterX + offX, afterY + offY, afterTetro, 1, False)
                        if parse is True:
                            parsed = True
                            judge(afterX + offX, afterY + offY, afterTetro)
                            if kick.index((offX, offY)) < 4 and type_c in spin_pic:
                                tmini = True
                                tspin = False
                            elif kick.index((offX, offY)) >= 4 and type_c in spin_pic:
                                tmini = False
                                tspin = True
                            else:
                                tmini = False
                                tspin = False
                            break
                        elif parse is False:
                            parse = True
                            tspin = False
                            tmini = False
                    if parsed is False:
                        afterTetro.clear()
                        afterTetro.extend(cout)
                        turn -= 1
                        if turn < 0:
                            turn += 4
                        dturn += 1
                        if dturn <= 2 and glob["doublet"] == "true":
                            dturn = 0
                            turn += 2
                            if turn > 3:
                                turn -= 4
                            cout.extend(afterTetro)
                            afterTetro.clear()
                            afterTetro.extend(tetro[type_c][turn])
                            for (offX, offY) in kick:
                                parse = judge(afterX + offX, afterY + offY, afterTetro, 1, False)
                                if parse is True:
                                    parsed = True
                                    judge(afterX + offX, afterY + offY, afterTetro)
                                    if kick.index((offX, offY)) < 4 and type_c in spin_pic:
                                        tmini = True
                                        tspin = False
                                    elif kick.index((offX, offY)) >= 4 and type_c in spin_pic:
                                        tmini = False
                                        tspin = True
                                    else:
                                        tmini = False
                                        tspin = False
                                    break
                                elif parse is False:
                                    parse = True
                                    tspin = False
                            if parsed is False:
                                afterTetro.clear()
                                afterTetro.extend(cout)
                                turn -= 2
                                if turn < 0:
                                    turn += 4
                        else:
                            afterTetro.clear()
                            afterTetro.extend(cout)
                            tspin = False
                            tmini = False
                        tspin = False
                        tmini = False
        elif event.keysym==cont["T_Left"] :
            spin = True
            cout = []
            turn_s.play()
            if hardd is False and pause is False:
                parse = True
                if type_c in even_pic:
                    kick = wallkickCC[turn+4]
                else:
                    kick = wallkickCC[turn]
                turn -= 1
                if turn < 0:
                    turn += 4
                cout.extend(afterTetro)
                afterTetro.clear()
                afterTetro.extend(tetro[type_c][turn])
                if type_c not in sym_pic:
                    for (offX, offY) in kick:
                        parse = judge(afterX + offX, afterY + offY, afterTetro, 1, False)
                        if parse is True:
                            parsed = True
                            judge(afterX + offX, afterY + offY, afterTetro)
                            if kick.index((offX, offY)) < 4 and type_c in spin_pic:
                                tmini = True
                                tspin = False
                            elif kick.index((offX, offY)) >= 4 and type_c in spin_pic:
                                tmini = False
                                tspin = True
                            else:
                                tmini = False
                                tspin = False
                            break
                        elif parse is False:
                            parse = True
                            tspin = False
                    if parsed is False:
                        afterTetro.clear()
                        afterTetro.extend(cout)
                        turn += 1
                        if turn > 3:
                            turn -= 4
                        dturn -= 1
                        if dturn >= -2 and glob["doublet"] == "true":
                            dturn = 0
                            turn -= 2
                            if turn < 0:
                                turn += 4
                            cout.extend(afterTetro)
                            afterTetro.clear()
                            afterTetro.extend(tetro[type_c][turn])
                            for (offX, offY) in kick:
                                parse = judge(afterX + offX, afterY + offY, afterTetro, 1, False)
                                if parse is True:
                                    parsed = True
                                    judge(afterX + offX, afterY + offY, afterTetro)
                                    if kick.index((offX, offY)) < 4 and type_c in spin_pic:
                                        tmini = True
                                        tspin = False
                                    elif kick.index((offX, offY)) >= 4 and type_c in spin_pic:
                                        tmini = False
                                        tspin = True
                                    else:
                                        tmini = False
                                        tspin = False
                                    break
                                elif parse is False:
                                    parse = True
                                    tspin = False
                            if parsed is False:
                                afterTetro.clear()
                                afterTetro.extend(cout)
                                turn += 2
                                if turn > 3:
                                    turn -= 4
                        else:
                            afterTetro.clear()
                            afterTetro.extend(cout)
                            tspin = False
                            tmini = False
                            
        elif event.keysym==cont["Hold"] :
            if pause is False:
                if glob["hold"] == "true":
                    if hcount == 1:
                        hold_s.play()
                        moveX = 5
                        moveY = 1
                        afterX = moveX
                        afterY = moveY
                        afterTetro.clear()
                        hcount -= 1
                        hardd = False
                        turn = 0
                        dturn = 0
                        spin = False
                        tmini = False
                        tspin = False
                        if type_hold == 100:
                            type_hold = type_c
                            type_c = type_next
                            type_next = type_0
                            type_0 = type_1
                            type_1 = type_2
                            type_2 = type_3
                            type_3 = type_4
                            type_4 = type_5
                            type_5 = RMG()
                            afterTetro = []
                            afterTetro.extend(tetro[type_c][turn])
                        elif type_hold != 100:
                            type_s = type_hold
                            type_hold = type_c
                            type_c = type_s
                            type_s = 100
                            afterTetro = []
                            afterTetro.extend(tetro[type_c][turn])
                             
def judge(afterX, afterY, afterTetro, offsetY = 0, expand = True): 
    global moveX, moveY, turn, add, move_s
    result = True
    for i in range(int(len(afterTetro)/2)):
        if type(afterTetro[i*2]) == float:
            add = 0.5
        else:
            add = 0
        x = int(afterTetro[i*2]+afterX + add)
        y = int(afterTetro[i*2+1]+afterY + add)
        try:
            if field[y+1][x]!=50 :
                result = False
        except IndexError:
            result = False
                
    if result==True and expand==True:
        moveX = afterX
        moveY = afterY+offsetY
    return result

def t_turn(turn):
    if turn == 0:
        return [0, 1]
    elif turn == 1:
        return [1, 3]
    elif turn == 2:
        return [2, 3]
    elif turn == 3:
        return [0, 2]
    
def tspin_perser(x, y, turn, tspin, tmini):
    if glob["rotation"] != "original" and glob["rotation"] != "sega" and glob["rotation"] != "left nrs" and glob["rotation"] != "right nrs" and glob["rotation"] != "ars":
        try:
            slot = [field[y][x-1],field[y][x+1],field[y+2][x-1],field[y+2][x+1]]
        except IndexError:
            if x+1 > 11 and y+2 > 21:
                slot = [field[y][x-1],0,0,0]
            elif x+1 > 11:
                slot = [field[y][x-1],0,field[y+2][x-1],0]
            elif y+2 > 21:
                slot = [field[y][x-1],field[y][x+1],0,0]
        rot = t_turn(turn)
        spe = True
        if slot.count(50) > 1:
            return "false"
        elif slot.count(50) == 1:
            for i in range(2):
                if slot[rot[i]] == 50:
                    spe = False
                    break
        if spe is True:
            if tspin is True or tmini is True:
                return "spin"
        elif spe is False:
            if tmini is True:
                return "mini"
        else:
            return "false"
    else:
        return "false"
    
def dropTetris(hard=False):
    global moveX, moveY, pc, add, replayed, scene, stone, snow, type_0, type_1, type_2, type_3, type_4, type_5, tspin, pause, spin, tmini, btb, dturn, turn, hcount, hardd, type_c, type_next, combo, hdrop, timer, rest_line, line, listed, score, level, hscore
    afterTetro = []
    chain = []
    field_a = []
    field_b = []
    afterTetro.extend(tetro[type_c][turn])
    chain.extend(tetro[type_c][turn])
    if snow != 0:
        timer = snow
        snow = 0
    if pause is True or scene == 0 or stone is True:
        result = judge(moveX, moveY, afterTetro)
    else:
        result = judge(moveX, moveY+1, afterTetro)
    if spin is False and (result==False or hard is True):
        field_a = field
        if type_c in even_pic:
            add = 0.5
        else:
            add = 0
        for i in range(int(len(tetro[type_c][turn])/2)):
            x = int(tetro[type_c][turn][i*2]+moveX+add)
            y = int(tetro[type_c][turn][i*2+1]+moveY+add)
            field[y+1][x] = type_c
            if type_c in spin_pic and spin is True:
                tspin_perse = tspin_perser(moveX, moveY+1, turn, tspin, tmini)
                if tspin_perse == "mini":
                    tmini = True
                    tspin = False
                elif tspin_perse == "spin":
                    tspin = True
                    tmini = False
                else:
                    tspin = False
                    tmini = False
        if hardd is False:
            land_s.play()
        hardd = False
        field_b = field
        deleteLine()
        afterTetro = []
        hcount = 1
        if field == field0:
            pc = True
            perfect_s.play()
        type_c = type_next
        type_next = type_0
        type_0 = type_1
        type_1 = type_2
        type_2 = type_3
        type_3 = type_4
        type_4 = type_5
        type_5 = RMG()
        moveX = 5
        moveY = 1
        turn = 0
        dturn = 0
            
...