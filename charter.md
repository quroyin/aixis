# QUANT PIPELINE CHARTER                            

## MAJOR PHASES (Architecture Layer)                                         

 1. Blueprint (PIPELINE_BLUEPRINT.md)  → WHERE to go (The Flow)           
 2. Schema (pipeline_schema.json)      → WHAT to expect (The Data)        
 3. Config (project_config.py)         → HOW to behave (The Settings)     

## MINOR PHASES (Quality Gates — must pass ALL before proceeding)            

 1. Decoupled?     Can I delete Phase N-1 and Phase N still runs?         
 2. Contract?      Does Phase N know exactly what Phase N-1 gives it?     
 3. Deterministic? Same inputs → same outputs, always?                    
 4. Atomic?        If power fails, is data safe or half-written?          
 5. Scalable?      Will it crash with 1000 stocks instead of 10?          
 6. Orchestrated?  Is there a main.py button to run everything?           
 7. Traceable?     Can I trace every output to its source inputs?         
 8. Versionable?   Can I mix Phase 2 v1.3 with Phase 1 v2.1?              
 9. Recoverable?   Can I resume Phase N without re-running Phase 1 to N-1?

## CODE STANDARDS (Every Python File Must Have)                              

 1. Auditor Prompt    Review logic before running (AI-assisted)           
 2. Runtime Manifest  Pickle checkpoints at phase boundaries              
 3. V-Report          Structured prints for debugging                     
 4. Idempotency       Safe to re-run without side effects                 
 5. Schema Enforcement Crash if data is wrong                             
 6. Memory Spooling   Don't crash on large datasets                       
 7. Fail-Fast         Crash immediately with clear errors                 
 8. Linearity         Read top-to-bottom like a story                     
