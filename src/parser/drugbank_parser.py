# src/ddi/parser/drugbank_parser.py

import os
import logging
import pickle
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
import gc
import time

import pandas as pd
from lxml import etree
from tqdm import tqdm

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula, CalcTPSA, CalcNumRotatableBonds, CalcNumHBD, CalcNumHBA, CalcNumAromaticRings
    from rdkit import RDLogger
    # Suppress RDKit warnings to reduce noise during processing
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

class DrugBankParser:
    """
    Enhanced DrugBank parser with comprehensive data extraction and memory optimization.
    Parses the main XML database and enriches it with vocabulary CSV data.
    """
    def __init__(self, xml_path: str, vocabulary_path: str, sdf_path: str, output_dir: str):
        self.xml_path = xml_path
        self.vocabulary_path = vocabulary_path
        self.sdf_path = sdf_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ns = {}  # XML namespace, to be detected automatically
        self.vocabulary_df = None
        self.sdf_data = {}  # Dictionary to store SDF molecular data
        self.integrated_data = None
        
        # Enhanced data containers
        self.parsing_stats = {
            'total_drugs': 0,
            'approved_drugs': 0,
            'experimental_drugs': 0,
            'drugs_with_targets': 0,
            'drugs_with_indications': 0,
            'drugs_with_structures': 0,
            'sdf_molecules_loaded': 0,
            'processing_time': 0.0
        }

    def parse_and_integrate(self, limit: Optional[int] = None, include_experimental: bool = True) -> None:
        """
        Memory-optimized enhanced orchestration method with comprehensive parsing options.
        Optimized for 16GB RAM systems with aggressive memory management.
        
        Args:
            limit: Optional limit on the number of drugs to process for testing.
            include_experimental: Whether to include experimental/investigational drugs.
        """
        start_time = time.time()
        
        self.logger.info("Starting enhanced DrugBank parsing (memory optimized for 16GB RAM)...")
        self.logger.info(f"Include experimental drugs: {include_experimental}")
        if limit:
            self.logger.info(f"Processing limit: {limit} drugs")

        # Memory monitoring setup
        self._log_memory_status("Initial memory state")

        # Step 1: Load the vocabulary CSV for enrichment
        self.logger.info("Step 1: Loading vocabulary data...")
        self._load_vocabulary()
        self._log_memory_status("After vocabulary loading")
        gc.collect()  # Clean up after vocabulary loading

        # Step 2: Load SDF structure data for chemical enrichment (memory efficient)
        self.logger.info("Step 2: Loading SDF structure data...")
        self._load_sdf_structures()
        self._log_memory_status("After SDF loading")
        gc.collect()  # Clean up after SDF loading

        # Step 3: Parse the main XML file with enhanced extraction (memory efficient)
        self.logger.info("Step 3: Parsing XML data...")
        version, parsed_drugs = self._parse_xml_enhanced(limit=limit, include_experimental=include_experimental)
        self._log_memory_status("After XML parsing")
        gc.collect()  # Clean up after XML parsing

        # Step 4: Integrate vocabulary and structure data into the parsed XML data (memory efficient)
        self.logger.info("Step 4: Integrating data sources...")
        self._integrate_data_enhanced(version, parsed_drugs)
        self._log_memory_status("After data integration")
        
        # Clear temporary data to free memory
        del parsed_drugs
        gc.collect()

        # Step 5: Validate and analyze the final integrated data
        self.logger.info("Step 5: Validating integrated data...")
        self._validate_and_analyze_data()
        
        self.parsing_stats['processing_time'] = time.time() - start_time
        self.logger.info(f"DrugBank parsing complete. Total time: {self.parsing_stats['processing_time']:.2f}s")
        self._log_parsing_statistics()
        self._log_memory_status("Final memory state")

    def _log_memory_status(self, stage: str) -> None:
        """Log current memory usage status with fallback when psutil is unavailable."""
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                self.logger.info(f"{stage}: RAM usage {memory.percent:.1f}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
                
                # Warning for high memory usage
                if memory.percent > 85:
                    self.logger.warning(f"High memory usage detected: {memory.percent:.1f}%")
                    gc.collect()  # Force cleanup
                    
            except Exception as e:
                self.logger.warning(f"{stage}: Memory monitoring failed: {e}")
        else:
            self.logger.info(f"{stage}: Memory monitoring unavailable (install psutil: pip install psutil)")

    def _load_vocabulary(self) -> None:
        """Loads and prepares the DrugBank vocabulary CSV."""
        self.logger.info(f"Loading DrugBank vocabulary from {self.vocabulary_path}")
        try:
            self.vocabulary_df = pd.read_csv(self.vocabulary_path)
            # Rename columns for consistency
            column_mapping = {
                "DrugBank ID": "drugbank_id", "Accession Numbers": "accession_numbers",
                "Common name": "name", "CAS": "cas_number", "UNII": "unii",
                "Synonyms": "synonyms", "Standard InChI Key": "inchikey"
            }
            self.vocabulary_df.rename(columns=column_mapping, inplace=True)
            # Set drugbank_id as the index for fast lookups
            self.vocabulary_df.set_index('drugbank_id', inplace=True)
            self.logger.info(f"Loaded {len(self.vocabulary_df)} entries from vocabulary.")
        except FileNotFoundError:
            self.logger.error(f"Vocabulary file not found: {self.vocabulary_path}. Proceeding without enrichment.")
            self.vocabulary_df = None
        except Exception as e:
            self.logger.error(f"Error loading vocabulary CSV: {e}. Proceeding without enrichment.")
            self.vocabulary_df = None

    def _load_sdf_structures(self) -> None:
        """Memory-efficient loading of DrugBank SDF structure file with aggressive garbage collection."""
        if not RDKIT_AVAILABLE:
            self.logger.warning("RDKit not available. Using basic SDF parsing.")
            self._load_sdf_structures_basic()
            return
            
        self.logger.info(f"Loading DrugBank SDF structures with RDKit from {self.sdf_path}")
        self.logger.info("Using memory-optimized processing for 16GB RAM system...")
        
        try:
            # Use RDKit's SDMolSupplier with memory optimization
            sdf_supplier = Chem.SDMolSupplier(self.sdf_path, removeHs=True, sanitize=True)  # Remove hydrogens to save memory
            sdf_count = 0
            batch_size = 500  # Process in smaller batches for memory efficiency
            batch_count = 0
            
            for mol in sdf_supplier:
                if mol is None:
                    continue
                    
                # Extract DrugBank ID from molecule properties
                drugbank_id = None
                prop_names = mol.GetPropNames()
                
                # Try different property names for DrugBank ID
                for prop_name in ['DATABASE_ID', 'DRUGBANK_ID', 'ID', 'drugbank_id']:
                    if mol.HasProp(prop_name):
                        drugbank_id = mol.GetProp(prop_name).strip()
                        break
                
                # If no explicit ID property, try the molecule name
                if not drugbank_id and mol.HasProp('_Name'):
                    drugbank_id = mol.GetProp('_Name').strip()
                
                if not drugbank_id:
                    continue
                
                # Calculate only essential molecular descriptors to save memory
                try:
                    molecular_data = {
                        'molecular_formula': CalcMolFormula(mol),
                        'exact_mass': Descriptors.ExactMolWt(mol),
                        'molecular_weight_rdkit': Descriptors.MolWt(mol),
                        'logp': Crippen.MolLogP(mol),
                        'tpsa': CalcTPSA(mol),
                        'rotatable_bonds': CalcNumRotatableBonds(mol),
                        'h_bond_donors': CalcNumHBD(mol),
                        'h_bond_acceptors': CalcNumHBA(mol),
                        'heavy_atoms': mol.GetNumHeavyAtoms(),
                        'canonical_smiles': Chem.MolToSmiles(mol, canonical=True),
                        'inchikey_rdkit': Chem.MolToInchiKey(mol)
                    }
                    
                    # Add optional descriptors only if memory allows
                    try:
                        if sdf_count < 8000:  # Limit detailed descriptors for memory
                            molecular_data['aromatic_rings'] = CalcNumAromaticRings(mol)
                            molecular_data['inchi'] = Chem.MolToInchi(mol)
                            molecular_data['num_rings'] = Descriptors.RingCount(mol)
                            molecular_data['num_heteroatoms'] = Descriptors.NumHeteroatoms(mol)
                    except (AttributeError, MemoryError):
                        pass  # Skip if not available or memory error
                    
                    # Add essential Lipinski's Rule of Five analysis only
                    molecular_data['lipinski_mw'] = Descriptors.MolWt(mol) <= 500
                    molecular_data['lipinski_logp'] = Crippen.MolLogP(mol) <= 5
                    molecular_data['lipinski_hbd'] = CalcNumHBD(mol) <= 5
                    molecular_data['lipinski_hba'] = CalcNumHBA(mol) <= 10
                    lipinski_violations = sum([
                        not molecular_data['lipinski_mw'],
                        not molecular_data['lipinski_logp'],
                        not molecular_data['lipinski_hbd'],
                        not molecular_data['lipinski_hba']
                    ])
                    molecular_data['lipinski_violations'] = lipinski_violations
                    molecular_data['lipinski_compliant'] = lipinski_violations <= 1
                    
                    # Store only essential SDF properties to save memory
                    essential_props = ['DRUGBANK_ID', 'DATABASE_ID', 'FORMULA', 'EXACT_MASS']
                    for prop_name in prop_names:
                        if prop_name.upper() in essential_props:
                            try:
                                prop_value = mol.GetProp(prop_name)
                                molecular_data[f'sdf_{prop_name.lower()}'] = prop_value
                            except:
                                pass
                    
                    self.sdf_data[drugbank_id] = molecular_data
                    sdf_count += 1
                    batch_count += 1
                    
                    # Aggressive memory management every batch
                    if batch_count >= batch_size:
                        gc.collect()  # Force garbage collection
                        batch_count = 0
                        
                        # Log progress with memory info
                        if sdf_count % 1000 == 0:
                            if PSUTIL_AVAILABLE:
                                try:
                                    memory_percent = psutil.virtual_memory().percent
                                    self.logger.info(f"Processed {sdf_count} SDF structures (RAM: {memory_percent:.1f}%)")
                                    
                                    # Emergency memory check
                                    if memory_percent > 85:
                                        self.logger.warning(f"High memory usage ({memory_percent:.1f}%), limiting further processing")
                                        break
                                except:
                                    self.logger.info(f"Processed {sdf_count} SDF structures...")
                            else:
                                self.logger.info(f"Processed {sdf_count} SDF structures...")
                        
                except Exception as e:
                    self.logger.warning(f"Error calculating RDKit descriptors for {drugbank_id}: {e}")
                    continue
                finally:
                    # Clear molecule reference immediately
                    del mol
            
            # Final garbage collection
            gc.collect()
            
            self.parsing_stats['sdf_molecules_loaded'] = sdf_count
            self.logger.info(f"Loaded {sdf_count} molecular structures with RDKit descriptors (memory optimized)")
            
        except FileNotFoundError:
            self.logger.error(f"SDF file not found: {self.sdf_path}. Proceeding without structure enrichment.")
            self.sdf_data = {}
        except MemoryError:
            self.logger.error("Memory error during SDF loading. Reducing dataset size...")
            gc.collect()
            # Keep partial data
        except Exception as e:
            self.logger.error(f"Error loading SDF file with RDKit: {e}. Trying basic parsing...")
            self._load_sdf_structures_basic()

    def _load_sdf_structures_basic(self) -> None:
        """Basic SDF parsing fallback when RDKit is not available."""
        self.logger.info(f"Loading DrugBank SDF structures (basic parsing) from {self.sdf_path}")
        try:
            sdf_count = 0
            current_molecule = {}
            current_id = None
            
            with open(self.sdf_path, 'r', encoding='utf-8') as sdf_file:
                lines = sdf_file.readlines()
                
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for molecule header (first line of each molecule)
                if line and not line.startswith('$$$$') and not line.startswith('>'):
                    # This could be a molecule name/ID line
                    current_id = line
                    current_molecule = {'molecule_name': line}
                    i += 1
                    continue
                
                # Look for property blocks (lines starting with >)
                if line.startswith('>'):
                    # Extract property name
                    prop_match = line.split('<')[1].split('>')[0] if '<' in line and '>' in line else None
                    if prop_match and i + 1 < len(lines):
                        prop_value = lines[i + 1].strip()
                        if prop_match.upper() in ['DATABASE_ID', 'DRUGBANK_ID', 'ID']:
                            current_id = prop_value
                        current_molecule[f'sdf_{prop_match.lower()}'] = prop_value
                    i += 2
                    continue
                
                # End of molecule marker
                if line == '$$$$':
                    if current_id and current_molecule:
                        # Try to extract actual DrugBank ID
                        drugbank_id = current_id
                        for key, value in current_molecule.items():
                            if 'drugbank' in key.lower() or 'database' in key.lower():
                                drugbank_id = value
                                break
                        
                        self.sdf_data[drugbank_id] = current_molecule
                        sdf_count += 1
                        
                        if sdf_count % 1000 == 0:
                            self.logger.info(f"Processed {sdf_count} SDF structures (basic)...")
                    
                    current_molecule = {}
                    current_id = None
                
                i += 1
            
            self.parsing_stats['sdf_molecules_loaded'] = sdf_count
            self.logger.info(f"Loaded {sdf_count} molecular structures (basic parsing)")
            
        except Exception as e:
            self.logger.error(f"Error in basic SDF parsing: {e}. Proceeding without structure enrichment.")
            self.sdf_data = {}

    def _parse_xml_enhanced(self, limit: Optional[int], include_experimental: bool = True) -> Tuple[str, List[Dict]]:
        """Memory-efficient enhanced DrugBank XML parsing with aggressive garbage collection."""
        self.logger.info(f"Parsing DrugBank XML file: {self.xml_path}")
        self.logger.info("Using memory-optimized parsing for 16GB RAM system...")
        
        drugs = []
        version = "unknown"
        drug_count = 0
        batch_size = 100  # Smaller batches for memory efficiency
        batch_count = 0
        
        try:
            # Use iterparse with minimal events for memory efficiency
            context = etree.iterparse(self.xml_path, events=("start", "end"), recover=True)
            
            for event, elem in context:
                # Detect namespace from the root element
                if event == 'start' and 'drugbank' in elem.tag and not self.ns:
                    if '}' in elem.tag and elem.tag.startswith('{'):
                        self.ns = {"db": elem.tag.split('}')[0][1:]}
                        version_elem = elem.find("db:version", self.ns)
                        if version_elem is not None:
                            version = version_elem.text.strip()
                        self.logger.info(f"Detected DrugBank Version: {version}, Namespace: {self.ns['db']}")

                # Process each 'drug' element at its end
                if event == 'end' and elem.tag == f"{{{self.ns.get('db')}}}drug":
                    try:
                        drug_dict = self._parse_drug_element_enhanced(elem, include_experimental)
                        if drug_dict:
                            drugs.append(drug_dict)
                            drug_count += 1
                            batch_count += 1
                            
                            # Update statistics
                            self.parsing_stats['total_drugs'] += 1
                            groups = drug_dict.get('groups', [])
                            if 'approved' in groups:
                                self.parsing_stats['approved_drugs'] += 1
                            if any(g in groups for g in ['experimental', 'investigational']):
                                self.parsing_stats['experimental_drugs'] += 1
                            if drug_dict.get('targets'):
                                self.parsing_stats['drugs_with_targets'] += 1
                            if drug_dict.get('indication'):
                                self.parsing_stats['drugs_with_indications'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"Error parsing a drug element: {e}", exc_info=True)
                    finally:
                        # Aggressive memory cleanup for each element
                        elem.clear()
                        while elem.getprevious() is not None:
                            del elem.getparent()[0]
                
                    # Batch-based memory management
                    if batch_count >= batch_size:
                        gc.collect()  # Force garbage collection every batch
                        batch_count = 0
                        
                        # Memory monitoring and logging
                        if drug_count % 1000 == 0:
                            if PSUTIL_AVAILABLE:
                                memory_percent = psutil.virtual_memory().percent
                                self.logger.info(f"Processed {drug_count} drugs (RAM: {memory_percent:.1f}%)")
                                
                                # Emergency memory management
                                if memory_percent > 85:
                                    self.logger.warning(f"High memory usage ({memory_percent:.1f}%), forcing aggressive cleanup")
                                    # Force more aggressive garbage collection
                                    gc.collect()
                                    gc.collect()  # Call twice for thorough cleanup
                                    
                                    # Emergency memory check
                                    memory_percent = psutil.virtual_memory().percent
                                    if memory_percent > 90:
                                        self.logger.error(f"Critical memory usage ({memory_percent:.1f}%), stopping processing")
                                        break
                            else:
                                # Fallback logging without psutil
                                self.logger.info(f"Processed {drug_count} drugs...")
                
                if limit and drug_count >= limit:
                    self.logger.info(f"Reached processing limit of {limit} drugs")
                    break

        except etree.XMLSyntaxError as e:
            self.logger.error(f"XML Syntax Error in {self.xml_path}: {e}")
        except FileNotFoundError:
            self.logger.error(f"XML file not found: {self.xml_path}")
        except MemoryError:
            self.logger.error("Memory error during XML parsing. System resources exhausted.")
            # Try to save what we have
            gc.collect()
        finally:
            # Final cleanup
            try:
                del context
            except:
                pass
            gc.collect()
        
        self.logger.info(f"Successfully parsed {len(drugs)} drugs from XML (memory optimized)")
        return version, drugs

    def _integrate_data_enhanced(self, version: str, drugs: List[Dict]):
        """Memory-efficient enhanced data integration with vocabulary and SDF structure enrichment."""
        self.logger.info("Integrating XML data with vocabulary and structure data (memory optimized)...")
        
        # Process in smaller chunks to manage memory
        chunk_size = 500  # Process 500 drugs at a time
        enriched_drugs = []
        vocab_enriched_count = 0
        structure_enriched_count = 0
        
        # Process drugs in chunks
        for i in range(0, len(drugs), chunk_size):
            chunk = drugs[i:i + chunk_size]
            chunk_enriched = []
            
            for drug in chunk:
                drug_id = drug.get("drugbank_id")
                
                # Vocabulary enrichment
                if self.vocabulary_df is not None and drug_id in self.vocabulary_df.index:
                    try:
                        vocab_series = self.vocabulary_df.loc[drug_id]
                        
                        # Enrich fields that are missing or empty in the XML data
                        if not drug.get("inchikey") and pd.notna(vocab_series.get("inchikey")):
                            drug["inchikey"] = vocab_series["inchikey"]
                        if not drug.get("unii") and pd.notna(vocab_series.get("unii")):
                            drug["unii"] = vocab_series["unii"]
                        if not drug.get("cas_number") and pd.notna(vocab_series.get("cas_number")):
                            drug["cas_number"] = vocab_series["cas_number"]
                            
                        # Combine synonyms (memory efficient)
                        xml_syns = set(drug.get("synonyms", []))
                        vocab_syns_raw = vocab_series.get("synonyms")
                        if pd.notna(vocab_syns_raw):
                            vocab_syns = set(str(vocab_syns_raw).split('|'))
                            combined_syns = xml_syns.union(vocab_syns)
                            drug["synonyms"] = sorted(list(combined_syns))
                            # Clear temporary sets
                            del xml_syns, vocab_syns, combined_syns
                            
                        vocab_enriched_count += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Error enriching drug {drug_id} with vocabulary: {e}")
                
                # SDF structure enrichment (memory efficient)
                if drug_id in self.sdf_data:
                    try:
                        sdf_data = self.sdf_data[drug_id]
                        
                        # Add only essential SDF properties to save memory
                        essential_keys = [
                            'molecular_formula', 'exact_mass', 'molecular_weight_rdkit', 
                            'logp', 'tpsa', 'canonical_smiles', 'inchikey_rdkit',
                            'lipinski_violations', 'lipinski_compliant'
                        ]
                        
                        for key in essential_keys:
                            if key in sdf_data and (key not in drug or not drug[key]):
                                drug[key] = sdf_data[key]
                        
                        # Add remaining properties only if memory allows
                        if structure_enriched_count < 8000:  # Limit detailed enrichment
                            for key, value in sdf_data.items():
                                if key not in essential_keys and key not in drug:
                                    drug[key] = value
                                    
                        # Mark that this drug has been enriched with structure data
                        drug['has_structure_data'] = True
                        structure_enriched_count += 1
                        self.parsing_stats['drugs_with_structures'] += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Error enriching drug {drug_id} with structure: {e}")
                        drug['has_structure_data'] = False
                else:
                    drug['has_structure_data'] = False
                    
                chunk_enriched.append(drug)
            
            # Add chunk to results and cleanup
            enriched_drugs.extend(chunk_enriched)
            del chunk, chunk_enriched
            
            # Garbage collection after each chunk
            gc.collect()
            
            # Progress logging with memory monitoring
            if (i + chunk_size) % 2000 == 0:
                if PSUTIL_AVAILABLE:
                    memory_percent = psutil.virtual_memory().percent
                    processed = min(i + chunk_size, len(drugs))
                    self.logger.info(f"Integrated {processed}/{len(drugs)} drugs (RAM: {memory_percent:.1f}%)")
                    
                    # Memory warning
                    if memory_percent > 85:
                        self.logger.warning(f"High memory usage during integration ({memory_percent:.1f}%)")
                        gc.collect()  # Extra cleanup
                else:
                    processed = min(i + chunk_size, len(drugs))
                    self.logger.info(f"Integrated {processed}/{len(drugs)} drugs")
        
        # Clear original drugs list and SDF data to free memory
        del drugs
        if hasattr(self, 'sdf_data'):
            # Keep only essential structure data if memory is tight
            if PSUTIL_AVAILABLE:
                if psutil.virtual_memory().percent > 80:
                    self.logger.info("High memory usage, clearing SDF cache...")
                    self.sdf_data.clear()
            else:
                # Conservative cleanup without memory monitoring
                if len(self.sdf_data) > 5000:
                    self.logger.info("Large SDF cache, clearing to save memory...")
                    self.sdf_data.clear()
        
        # Final garbage collection
        gc.collect()
        
        self.integrated_data = enriched_drugs
        self.logger.info(f"Finished integration. Total drugs: {len(self.integrated_data)}")
        self.logger.info(f"Enriched {vocab_enriched_count} drugs with vocabulary data")
        self.logger.info(f"Enriched {structure_enriched_count} drugs with SDF structure data")

    def _validate_and_analyze_data(self) -> None:
        """Enhanced validation and analysis of the final integrated data."""
        if not self.integrated_data:
            self.logger.warning("No integrated data to validate.")
            return

        num_drugs = len(self.integrated_data)
        missing_fields = {
            "drugbank_id": 0, "name": 0, "inchikey": 0, "indication": 0,
            "targets": 0, "synonyms": 0, "mechanism_of_action": 0
        }
        
        group_counts = {}
        state_counts = {}
        structure_enriched = 0
        
        for drug in self.integrated_data:
            # Count missing fields
            for field in missing_fields:
                if not drug.get(field):
                    missing_fields[field] += 1
                    
            # Analyze drug groups
            for group in drug.get('groups', []):
                group_counts[group] = group_counts.get(group, 0) + 1
                
            # Analyze drug states
            state = drug.get('state') or 'unknown'  # Handle None values
            state_counts[state] = state_counts.get(state, 0) + 1
            
            # Count structure-enriched drugs
            if drug.get('has_structure_data', False):
                structure_enriched += 1
        
        self.logger.info("Enhanced data validation results:")
        self.logger.info(f"Total drugs processed: {num_drugs}")
        
        for field, count in missing_fields.items():
            percentage = (count / num_drugs) * 100
            self.logger.info(f"  - Drugs missing '{field}': {count} ({percentage:.2f}%)")
            
        self.logger.info("Drug group distribution:")
        for group, count in sorted(group_counts.items()):
            percentage = (count / num_drugs) * 100
            self.logger.info(f"  - {group}: {count} ({percentage:.2f}%)")
            
        self.logger.info("Drug state distribution:")
        for state, count in sorted(state_counts.items()):
            percentage = (count / num_drugs) * 100
            self.logger.info(f"  - {state}: {count} ({percentage:.2f}%)")
            
        self.logger.info("Structure data enrichment:")
        structure_percentage = (structure_enriched / num_drugs) * 100
        self.logger.info(f"  - Drugs with SDF structure data: {structure_enriched} ({structure_percentage:.2f}%)")

    def _log_parsing_statistics(self):
        """Log comprehensive parsing statistics."""
        self.logger.info("=== DRUGBANK PARSING STATISTICS ===")
        for key, value in self.parsing_stats.items():
            if key == 'processing_time':
                self.logger.info(f"{key.replace('_', ' ').title()}: {value:.2f} seconds")
            else:
                self.logger.info(f"{key.replace('_', ' ').title()}: {value}")
        self.logger.info("=" * 37)

    def _parse_xml(self, limit: Optional[int]) -> Tuple[str, List[Dict]]:
        """Parses the DrugBank XML file iteratively using lxml."""
        self.logger.info(f"Parsing DrugBank XML file: {self.xml_path}")
        drugs = []
        version = "unknown"
        try:
            # <<< THE FIX IS HERE: Changed 'self.xml_file_path' to 'self.xml_path'
            context = etree.iterparse(self.xml_path, events=("start", "end"), recover=True)
            drug_count = 0
            
            for event, elem in context:
                # Detect namespace from the root element
                if event == 'start' and 'drugbank' in elem.tag and not self.ns:
                    if '}' in elem.tag and elem.tag.startswith('{'):
                        self.ns = {"db": elem.tag.split('}')[0][1:]}
                        version_elem = elem.find("db:version", self.ns)
                        if version_elem is not None:
                            version = version_elem.text.strip()
                        self.logger.info(f"Detected DrugBank Version: {version}, Namespace: {self.ns['db']}")

                # Process each 'drug' element at its end
                if event == 'end' and elem.tag == f"{{{self.ns.get('db')}}}drug":
                    try:
                        drug_dict = self._parse_drug_element(elem)
                        if drug_dict:
                            drugs.append(drug_dict)
                            drug_count += 1
                    except Exception as e:
                        self.logger.error(f"Error parsing a drug element: {e}", exc_info=True)
                    
                    # Clear element for memory efficiency
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                
                if limit and drug_count >= limit:
                    break

        except etree.XMLSyntaxError as e:
            self.logger.error(f"XML Syntax Error in {self.xml_path}: {e}")
        except FileNotFoundError:
            self.logger.error(f"XML file not found: {self.xml_path}")
        
        self.logger.info(f"Successfully parsed {len(drugs)} drugs from XML.")
        return version, drugs

    def _integrate_data(self, version: str, drugs: List[Dict]):
        """Enriches parsed drug data with information from the vocabulary."""
        if self.vocabulary_df is None:
            self.logger.warning("Vocabulary not loaded. Skipping integration.")
            self.integrated_data = drugs
            return
        
        self.logger.info("Integrating XML data with vocabulary...")
        enriched_drugs = []
        for drug in tqdm(drugs, desc="Enriching drug data"):
            drug_id = drug.get("drugbank_id")
            if drug_id in self.vocabulary_df.index:
                vocab_series = self.vocabulary_df.loc[drug_id]
                # Enrich fields that are missing or empty in the XML data
                if not drug.get("inchikey") and pd.notna(vocab_series.get("inchikey")):
                    drug["inchikey"] = vocab_series["inchikey"]
                if not drug.get("unii") and pd.notna(vocab_series.get("unii")):
                    drug["unii"] = vocab_series["unii"]
                # Combine synonyms
                xml_syns = set(drug.get("synonyms", []))
                vocab_syns_raw = vocab_series.get("synonyms")
                if pd.notna(vocab_syns_raw):
                    vocab_syns = set(vocab_syns_raw.split('|'))
                    drug["synonyms"] = sorted(list(xml_syns.union(vocab_syns)))
            enriched_drugs.append(drug)
        
        # The final processed data is just the list of drugs
        self.integrated_data = enriched_drugs
        self.logger.info(f"Finished integration. Total drugs: {len(self.integrated_data)}")

    def _validate_data(self) -> None:
        """Performs a simple validation on the final integrated data."""
        if not self.integrated_data:
            self.logger.warning("No integrated data to validate.")
            return

        num_drugs = len(self.integrated_data)
        missing_fields = {"drugbank_id": 0, "name": 0, "inchikey": 0}
        
        for drug in self.integrated_data:
            for field in missing_fields:
                if not drug.get(field):
                    missing_fields[field] += 1
        
        self.logger.info("Data validation results:")
        for field, count in missing_fields.items():
            percentage = (count / num_drugs) * 100
            self.logger.info(f"  - Drugs missing '{field}': {count} ({percentage:.2f}%)")

    def save(self, format: str = "pickle") -> Optional[str]:
        """Memory-efficient saving of the final integrated data to a file."""
        if not self.integrated_data:
            self.logger.warning("No integrated data to save.")
            return None
        
        # We save just the list of drugs, not the wrapper dictionary
        output_path = os.path.join(self.output_dir, f"drugbank_parsed.{format}")
        self.logger.info(f"Saving {len(self.integrated_data)} drugs to {output_path} (memory optimized)...")

        # Log memory before saving
        self._log_memory_status("Before saving data")

        try:
            if format == "pickle":
                # Use protocol 4 for better compression and memory efficiency
                with open(output_path, "wb") as f:
                    pickle.dump(self.integrated_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif format == "json":
                # Save in chunks for large datasets to avoid memory issues
                chunk_size = 1000
                self.logger.info(f"Saving JSON in chunks of {chunk_size} for memory efficiency...")
                
                with open(output_path, "w") as f:
                    f.write("[\n")
                    for i in range(0, len(self.integrated_data), chunk_size):
                        chunk = self.integrated_data[i:i + chunk_size]
                        
                        for j, drug in enumerate(chunk):
                            if i + j > 0:  # Add comma for all except first item
                                f.write(",\n")
                            json.dump(drug, f, indent=2)
                        
                        # Force memory cleanup after each chunk
                        del chunk
                        gc.collect()
                        
                        if (i + chunk_size) % 5000 == 0:
                            self.logger.info(f"Saved {min(i + chunk_size, len(self.integrated_data))}/{len(self.integrated_data)} drugs to JSON...")
                    
                    f.write("\n]")
            else:
                self.logger.error(f"Unsupported format: {format}")
                return None
            
            # Log memory after saving
            self._log_memory_status("After saving data")
            
            self.logger.info(f"Successfully saved data to {output_path}")
            return output_path
            
        except MemoryError:
            self.logger.error("Memory error during saving. Try reducing the dataset size or use pickle format.")
            return None
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            return None

    # --- Helper methods for parsing specific XML elements ---
    # These are consolidated from the original xml_parser.py
    
    def _get_text(self, element: etree._Element, xpath: str) -> Optional[str]:
        """Safely gets text from a sub-element."""
        target = element.find(xpath, self.ns)
        return target.text.strip() if target is not None and target.text else None

    def _parse_list(self, element: etree._Element, xpath: str) -> List[str]:
        """Parses a list of simple text elements."""
        items = []
        for item in element.findall(xpath, self.ns):
            if item.text:
                items.append(item.text.strip())
        return items

    def _parse_drug_element_enhanced(self, drug_elem: etree._Element, include_experimental: bool = True) -> Optional[Dict[str, Any]]:
        """Enhanced parsing of a single <drug> element with comprehensive data extraction."""
        # Get the primary ID first
        primary_id_elem = drug_elem.find("db:drugbank-id[@primary='true']", self.ns)
        drug_id = primary_id_elem.text.strip() if primary_id_elem is not None else self._get_text(drug_elem, "db:drugbank-id")
        
        if not drug_id:
            return None

        # Extract drug groups to filter experimental drugs if needed
        groups = self._parse_list(drug_elem, "db:groups/db:group")
        if not include_experimental and any(g in groups for g in ['experimental', 'investigational']):
            return None

        # Extract calculated properties for InChIKey and other properties
        inchikey = None
        molecular_weight = None
        smiles = None
        calc_props = drug_elem.find("db:calculated-properties", self.ns)
        if calc_props is not None:
            for prop in calc_props.findall("db:property", self.ns):
                kind = self._get_text(prop, "db:kind")
                value = self._get_text(prop, "db:value")
                if kind == "InChIKey":
                    inchikey = value
                elif kind == "Molecular Weight":
                    molecular_weight = value
                elif kind == "SMILES":
                    smiles = value

        # Enhanced ATC codes extraction
        atc_codes = []
        atc_elem = drug_elem.find("db:atc-codes", self.ns)
        if atc_elem is not None:
            for atc in atc_elem.findall("db:atc-code", self.ns):
                atc_codes.append(self._get_text(atc, "db:code"))

        # Enhanced categories extraction
        categories = []
        cat_elem = drug_elem.find("db:categories", self.ns)
        if cat_elem is not None:
            for cat in cat_elem.findall("db:category", self.ns):
                categories.append(self._get_text(cat, "db:category"))

        # Enhanced pathways extraction
        pathways = []
        pathways_elem = drug_elem.find("db:pathways", self.ns)
        if pathways_elem is not None:
            for pathway in pathways_elem.findall("db:pathway", self.ns):
                pathway_dict = {
                    "smpdb_id": self._get_text(pathway, "db:smpdb-id"),
                    "name": self._get_text(pathway, "db:name"),
                    "category": self._get_text(pathway, "db:category")
                }
                pathways.append(pathway_dict)

        return {
            "drugbank_id": drug_id,
            "name": self._get_text(drug_elem, "db:name"),
            "description": self._get_text(drug_elem, "db:description"),
            "cas_number": self._get_text(drug_elem, "db:cas-number"),
            "unii": self._get_text(drug_elem, "db:unii"),
            "inchikey": inchikey,
            "molecular_weight": molecular_weight,
            "smiles": smiles,
            "state": self._get_text(drug_elem, "db:state"),
            "groups": groups,
            "atc_codes": atc_codes,
            "categories": categories,
            "indication": self._get_text(drug_elem, "db:indication"),
            "pharmacodynamics": self._get_text(drug_elem, "db:pharmacodynamics"),
            "mechanism_of_action": self._get_text(drug_elem, "db:mechanism-of-action"),
            "toxicity": self._get_text(drug_elem, "db:toxicity"),
            "metabolism": self._get_text(drug_elem, "db:metabolism"),
            "absorption": self._get_text(drug_elem, "db:absorption"),
            "half_life": self._get_text(drug_elem, "db:half-life"),
            "protein_binding": self._get_text(drug_elem, "db:protein-binding"),
            "route_of_elimination": self._get_text(drug_elem, "db:route-of-elimination"),
            "synonyms": self._parse_list(drug_elem, "db:synonyms/db:synonym"),
            "pathways": pathways,
            "targets": self._parse_protein_relations(drug_elem, "db:targets/db:target"),
            "enzymes": self._parse_protein_relations(drug_elem, "db:enzymes/db:enzyme"),
            "transporters": self._parse_protein_relations(drug_elem, "db:transporters/db:transporter"),
            "carriers": self._parse_protein_relations(drug_elem, "db:carriers/db:carrier"),
        }

    def _parse_drug_element(self, drug_elem: etree._Element) -> Optional[Dict[str, Any]]:
        """Parses a single <drug> element into a dictionary."""
        # Get the primary ID first
        primary_id_elem = drug_elem.find("db:drugbank-id[@primary='true']", self.ns)
        drug_id = primary_id_elem.text.strip() if primary_id_elem is not None else self._get_text(drug_elem, "db:drugbank-id")
        
        if not drug_id:
            return None

        # Extract calculated properties for InChIKey
        inchikey = None
        calc_props = drug_elem.find("db:calculated-properties", self.ns)
        if calc_props is not None:
            for prop in calc_props.findall("db:property", self.ns):
                kind = self._get_text(prop, "db:kind")
                if kind == "InChIKey":
                    inchikey = self._get_text(prop, "db:value")
                    break

        return {
            "drugbank_id": drug_id,
            "name": self._get_text(drug_elem, "db:name"),
            "description": self._get_text(drug_elem, "db:description"),
            "cas_number": self._get_text(drug_elem, "db:cas-number"),
            "unii": self._get_text(drug_elem, "db:unii"),
            "inchikey": inchikey,
            "state": self._get_text(drug_elem, "db:state"),
            "groups": self._parse_list(drug_elem, "db:groups/db:group"),
            "indication": self._get_text(drug_elem, "db:indication"),
            "pharmacodynamics": self._get_text(drug_elem, "db:pharmacodynamics"),
            "mechanism_of_action": self._get_text(drug_elem, "db:mechanism-of-action"),
            "synonyms": self._parse_list(drug_elem, "db:synonyms/db:synonym"),
            "targets": self._parse_protein_relations(drug_elem, "db:targets/db:target"),
            "enzymes": self._parse_protein_relations(drug_elem, "db:enzymes/db:enzyme"),
            "transporters": self._parse_protein_relations(drug_elem, "db:transporters/db:transporter"),
            "carriers": self._parse_protein_relations(drug_elem, "db:carriers/db:carrier"),
        }

    def _parse_protein_relations(self, drug_elem: etree._Element, xpath: str) -> List[Dict]:
        """Parses protein relationships like targets, enzymes, etc."""
        relations = []
        for protein_elem in drug_elem.findall(xpath, self.ns):
            polypeptide = protein_elem.find("db:polypeptide", self.ns)
            uniprot_id = None
            if polypeptide is not None:
                ext_ids = polypeptide.find("db:external-identifiers", self.ns)
                if ext_ids is not None:
                    for ext_id in ext_ids.findall("db:external-identifier", self.ns):
                        if self._get_text(ext_id, "db:resource") == "UniProtKB":
                            uniprot_id = self._get_text(ext_id, "db:identifier")
                            break
            
            relation = {
                "id": self._get_text(protein_elem, "db:id"),
                "name": self._get_text(protein_elem, "db:name"),
                "organism": self._get_text(protein_elem, "db:organism"),
                "actions": self._parse_list(protein_elem, "db:actions/db:action"),
                "uniprot_id": uniprot_id
            }
            relations.append(relation)
        return relations


# --- Main execution block to make the script runnable ---
def main():
    parser = argparse.ArgumentParser(description="Enhanced DrugBank parser for comprehensive data extraction from XML, CSV vocabulary, and SDF structure files.")
    parser.add_argument("--xml", required=True, help="Path to the DrugBank full_database.xml file.")
    parser.add_argument("--vocab", required=True, help="Path to the drugbank_vocabulary.csv file.")
    parser.add_argument("--sdf", required=True, help="Path to the drugbank SDF structure file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the processed output file.")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle", help="Output file format.")
    parser.add_argument("--limit", type=int, help="Optional: limit the number of drugs to parse for testing.")
    parser.add_argument("--include_experimental", action="store_true", default=True, 
                        help="Include experimental/investigational drugs (default: True)")
    parser.add_argument("--approved_only", action="store_true", 
                        help="Process only approved drugs (excludes experimental)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Determine experimental inclusion based on arguments
    include_experimental = not args.approved_only if args.approved_only else args.include_experimental
    
    # Initialize and run the enhanced parser
    db_parser = DrugBankParser(
        xml_path=args.xml,
        vocabulary_path=args.vocab,
        sdf_path=args.sdf,
        output_dir=args.output_dir
    )
    db_parser.parse_and_integrate(limit=args.limit, include_experimental=include_experimental)
    db_parser.save(format=args.format)

if __name__ == "__main__":
    main()
