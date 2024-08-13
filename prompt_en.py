instruct_CST = """Given an entire context as the Context, generate a Question about the entire context that users might be interested in, which answer should be able to be derived directly from the Context. Then, divide the entire context into two sub-contexts Context 1 and Context 2 based on their semantic content, making necessary adjustments within each sub-contexts to ensure they are independently coherent.

---

Provide in the following form:

Context: The entire context

Question: Regarding the entire context

Context 1: Sub-context 1

Context 2: Sub-context 2

---

Context: Trying to rebound from their divisional home loss to the Buccaneers, the Panthers flew to the Louisiana Superdome for a Week 5 divisional duel with the winless New Orleans Saints. With QB Jake Delhomme out and done for the year with a right elbow injury, QB David Carr was given the start. In the first quarter, Carolina took the early lead with kicker John Kasay getting a 23-yard field goal. The Saints responded with kicker Olindo Mare getting a 25-yard field goal. In the second quarter, the Panthers went back into the lead with Kasay nailing a 35-yard field goal. New Orleans would respond with Mare kicking a 28-yard field goal. In the third quarter, Carolina trailed as Saints FB Mike Karney got a 2-yard TD run for the only score of the period. In the fourth quarter, the Panthers tied the game with Carr completing a 17-yard TD pass to WR Steve Smith. Afterwards, Carolina sealed the win in the final seconds with Kasay nailing a 52-yard field goal as time ran out.

Question: How did the Carolina Panthers secure their victory against the New Orleans Saints in their Week 5 divisional duel?

Context 1: Trying to rebound from their divisional home loss to the Buccaneers, the Panthers flew to the Louisiana Superdome for a Week 5 divisional duel with the winless New Orleans Saints. With QB Jake Delhomme out for the year with a right elbow injury, QB David Carr was given the start. In the first quarter, Carolina took the early lead with kicker John Kasay getting a 23-yard field goal. The Saints responded with kicker Olindo Mare getting a 25-yard field goal. In the second quarter, the Panthers went back into the lead with Kasay nailing a 35-yard field goal, followed by New Orleans’ response with Mare kicking a 28-yard field goal.

Context 2: As the game progressed into the third quarter, the Panthers found themselves trailing after Saints FB Mike Karney got a 2-yard TD run, marking the only score of the period. However, in the fourth quarter, the Panthers managed to tie the game thanks to QB David Carr completing a 17-yard TD pass to WR Steve Smith. The climax of the match came in the final seconds with John Kasay nailing a 52-yard field goal as time ran out, securing a dramatic victory for Carolina against the New Orleans Saints.

---

Context: As a cell grows, its volume increases more quickly than its surface area. If a cell was to get very large, the small surface area would not allow enough nutrients to enter the cell quickly enough for the cell’s needs. However, large cells have a way of dealing with some size challenges. Big cells, such as some white blood cells, often grow more nuclei so that they can supply enough proteins and RNA for the cell’s requirements. Large, metabolically active cells often have lots of cell protrusions, resulting in many folds throughout the membrane. These folds increase the surface area available for transport of materials into or out of the cell. Such cell types are found lining your small intestine, where they absorb nutrients from your food through protrusions called microvilli.

Question: How do large cells adapt to the challenge of having a volume that increases more quickly than their surface area to meet their metabolic needs?

Context 1: As a cell grows, its volume increases more quickly than its surface area. If a cell was to get very large, the small surface area would not allow enough nutrients to enter the cell quickly enough for the cell's needs.

Context 2: Large cells have a way of dealing with their size challenges. Big cells, such as some white blood cells, often grow more nuclei so that they can supply enough proteins and RNA for the cell's requirements. Large, metabolically active cells often have lots of cell protrusions, resulting in many folds throughout the membrane. These folds increase the surface area available for transport of materials into or out of the cell. Such cell types are found lining your small intestine, where they absorb nutrients from your food through protrusions called microvilli.


---

Context: Philip Arnold Heseltine is best known as a composer of songs and other vocal music; he also achieved notoriety in his lifetime through his unconventional and often scandalous lifestyle.

Question: Why is Philip Arnold Heseltine's reputation mixed?

Context 1: Philip Arnold Heseltine is best known as a composer of songs and other vocal music.

Context 2: Philip Arnold Heseltine also achieved notoriety in his lifetime through his unconventional and often scandalous lifestyle.


---

Context: {context}

Question: """


instruct_CST_instr = """Given a context, generate a question and split context into two sub-contexts.

---

Provide in the following form:

Context: The entire context

Question: Regarding the entire context

Context 1: Sub-context 1

Context 2: Sub-context 2

---

Context: Trying to rebound from their divisional home loss to the Buccaneers, the Panthers flew to the Louisiana Superdome for a Week 5 divisional duel with the winless New Orleans Saints. With QB Jake Delhomme out and done for the year with a right elbow injury, QB David Carr was given the start. In the first quarter, Carolina took the early lead with kicker John Kasay getting a 23-yard field goal. The Saints responded with kicker Olindo Mare getting a 25-yard field goal. In the second quarter, the Panthers went back into the lead with Kasay nailing a 35-yard field goal. New Orleans would respond with Mare kicking a 28-yard field goal. In the third quarter, Carolina trailed as Saints FB Mike Karney got a 2-yard TD run for the only score of the period. In the fourth quarter, the Panthers tied the game with Carr completing a 17-yard TD pass to WR Steve Smith. Afterwards, Carolina sealed the win in the final seconds with Kasay nailing a 52-yard field goal as time ran out.

Question: How did the Carolina Panthers secure their victory against the New Orleans Saints in their Week 5 divisional duel?

Context 1: Trying to rebound from their divisional home loss to the Buccaneers, the Panthers flew to the Louisiana Superdome for a Week 5 divisional duel with the winless New Orleans Saints. With QB Jake Delhomme out for the year with a right elbow injury, QB David Carr was given the start. In the first quarter, Carolina took the early lead with kicker John Kasay getting a 23-yard field goal. The Saints responded with kicker Olindo Mare getting a 25-yard field goal. In the second quarter, the Panthers went back into the lead with Kasay nailing a 35-yard field goal, followed by New Orleans’ response with Mare kicking a 28-yard field goal.

Context 2: As the game progressed into the third quarter, the Panthers found themselves trailing after Saints FB Mike Karney got a 2-yard TD run, marking the only score of the period. However, in the fourth quarter, the Panthers managed to tie the game thanks to QB David Carr completing a 17-yard TD pass to WR Steve Smith. The climax of the match came in the final seconds with John Kasay nailing a 52-yard field goal as time ran out, securing a dramatic victory for Carolina against the New Orleans Saints.

---

Context: As a cell grows, its volume increases more quickly than its surface area. If a cell was to get very large, the small surface area would not allow enough nutrients to enter the cell quickly enough for the cell’s needs. However, large cells have a way of dealing with some size challenges. Big cells, such as some white blood cells, often grow more nuclei so that they can supply enough proteins and RNA for the cell’s requirements. Large, metabolically active cells often have lots of cell protrusions, resulting in many folds throughout the membrane. These folds increase the surface area available for transport of materials into or out of the cell. Such cell types are found lining your small intestine, where they absorb nutrients from your food through protrusions called microvilli.

Question: How do large cells adapt to the challenge of having a volume that increases more quickly than their surface area to meet their metabolic needs?

Context 1: As a cell grows, its volume increases more quickly than its surface area. If a cell was to get very large, the small surface area would not allow enough nutrients to enter the cell quickly enough for the cell's needs.

Context 2: Large cells have a way of dealing with their size challenges. Big cells, such as some white blood cells, often grow more nuclei so that they can supply enough proteins and RNA for the cell's requirements. Large, metabolically active cells often have lots of cell protrusions, resulting in many folds throughout the membrane. These folds increase the surface area available for transport of materials into or out of the cell. Such cell types are found lining your small intestine, where they absorb nutrients from your food through protrusions called microvilli.


---

Context: Philip Arnold Heseltine is best known as a composer of songs and other vocal music; he also achieved notoriety in his lifetime through his unconventional and often scandalous lifestyle.

Question: Why is Philip Arnold Heseltine's reputation mixed?

Context 1: Philip Arnold Heseltine is best known as a composer of songs and other vocal music.

Context 2: Philip Arnold Heseltine also achieved notoriety in his lifetime through his unconventional and often scandalous lifestyle.


---

Context: {context}

Question: """


instruct_CST_fs = """Given an entire context as the Context, generate a Question about the entire context that users might be interested in, which answer should be able to be derived directly from the Context. Then, divide the entire context into two sub-contexts Context 1 and Context 2 based on their semantic content, making necessary adjustments within each sub-contexts to ensure they are independently coherent.

---

Provide in the following form:

Context: The entire context

Question: Regarding the entire context

Context 1: Sub-context 1

Context 2: Sub-context 2

---

Context: Trying to rebound from their divisional home loss to the Buccaneers, the Panthers flew to the Louisiana Superdome for a Week 5 divisional duel with the winless New Orleans Saints. With QB Jake Delhomme out and done for the year with a right elbow injury, QB David Carr was given the start. In the first quarter, Carolina took the early lead with kicker John Kasay getting a 23-yard field goal. The Saints responded with kicker Olindo Mare getting a 25-yard field goal. In the second quarter, the Panthers went back into the lead with Kasay nailing a 35-yard field goal. New Orleans would respond with Mare kicking a 28-yard field goal. In the third quarter, Carolina trailed as Saints FB Mike Karney got a 2-yard TD run for the only score of the period. In the fourth quarter, the Panthers tied the game with Carr completing a 17-yard TD pass to WR Steve Smith. Afterwards, Carolina sealed the win in the final seconds with Kasay nailing a 52-yard field goal as time ran out.

Question: How did the Carolina Panthers secure their victory against the New Orleans Saints in their Week 5 divisional duel?

Context 1: Trying to rebound from their divisional home loss to the Buccaneers, the Panthers flew to the Louisiana Superdome for a Week 5 divisional duel with the winless New Orleans Saints. With QB Jake Delhomme out for the year with a right elbow injury, QB David Carr was given the start. In the first quarter, Carolina took the early lead with kicker John Kasay getting a 23-yard field goal. The Saints responded with kicker Olindo Mare getting a 25-yard field goal. In the second quarter, the Panthers went back into the lead with Kasay nailing a 35-yard field goal, followed by New Orleans’ response with Mare kicking a 28-yard field goal.

Context 2: As the game progressed into the third quarter, the Panthers found themselves trailing after Saints FB Mike Karney got a 2-yard TD run, marking the only score of the period. However, in the fourth quarter, the Panthers managed to tie the game thanks to QB David Carr completing a 17-yard TD pass to WR Steve Smith. The climax of the match came in the final seconds with John Kasay nailing a 52-yard field goal as time ran out, securing a dramatic victory for Carolina against the New Orleans Saints.

---

Context: {context}

Question: """

instruct_CST_instr_fs = """Given a context, generate a question and split context into two sub-contexts.

---

Provide in the following form:

Context: The entire context

Question: Regarding the entire context

Context 1: Sub-context 1

Context 2: Sub-context 2

---

Context: Trying to rebound from their divisional home loss to the Buccaneers, the Panthers flew to the Louisiana Superdome for a Week 5 divisional duel with the winless New Orleans Saints. With QB Jake Delhomme out and done for the year with a right elbow injury, QB David Carr was given the start. In the first quarter, Carolina took the early lead with kicker John Kasay getting a 23-yard field goal. The Saints responded with kicker Olindo Mare getting a 25-yard field goal. In the second quarter, the Panthers went back into the lead with Kasay nailing a 35-yard field goal. New Orleans would respond with Mare kicking a 28-yard field goal. In the third quarter, Carolina trailed as Saints FB Mike Karney got a 2-yard TD run for the only score of the period. In the fourth quarter, the Panthers tied the game with Carr completing a 17-yard TD pass to WR Steve Smith. Afterwards, Carolina sealed the win in the final seconds with Kasay nailing a 52-yard field goal as time ran out.

Question: How did the Carolina Panthers secure their victory against the New Orleans Saints in their Week 5 divisional duel?

Context 1: Trying to rebound from their divisional home loss to the Buccaneers, the Panthers flew to the Louisiana Superdome for a Week 5 divisional duel with the winless New Orleans Saints. With QB Jake Delhomme out for the year with a right elbow injury, QB David Carr was given the start. In the first quarter, Carolina took the early lead with kicker John Kasay getting a 23-yard field goal. The Saints responded with kicker Olindo Mare getting a 25-yard field goal. In the second quarter, the Panthers went back into the lead with Kasay nailing a 35-yard field goal, followed by New Orleans’ response with Mare kicking a 28-yard field goal.

Context 2: As the game progressed into the third quarter, the Panthers found themselves trailing after Saints FB Mike Karney got a 2-yard TD run, marking the only score of the period. However, in the fourth quarter, the Panthers managed to tie the game thanks to QB David Carr completing a 17-yard TD pass to WR Steve Smith. The climax of the match came in the final seconds with John Kasay nailing a 52-yard field goal as time ran out, securing a dramatic victory for Carolina against the New Orleans Saints.

---

Context: {context}

Question: """

neg_instruct_CST_list = [instruct_CST_instr, instruct_CST_fs, instruct_CST_instr_fs]
