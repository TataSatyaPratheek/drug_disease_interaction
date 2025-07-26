**1. association_by_datasource_direct**

This table represents associations between diseases and targets, as calculated directly from individual data sources—without applying ontology propagation (i.e., no extra inference or relationship expansion from related entities). Each entry in this table uses the following key fields:

- **datatypeId**: A string representing the type of data used for scoring the association (e.g., genetic evidence, text mining, etc.).
- **datasourceId**: A string identifying the specific data source contributing the evidence (e.g., ChEMBL for chemistry, GWAS Catalog for genetics).
- **diseaseId**: The unique string identifier for the disease in question.
- **targetId**: The unique string identifier for the target (often a gene or protein).
- **score**: A numeric value quantifying the strength of the association, as calculated by this particular data source and datatype alone.
- **evidenceCount**: A numeric value indicating how many pieces of direct evidence support this association from the datasource.

This schema is the most granular view, assigning association scores based on independent evidence coming from each unique source and data type combination. The lack of ontology propagation means only direct links are considered, allowing precise attribution of evidence.

**2. association_by_datatype_direct**

This table focuses on direct associations but groups evidence by data type instead of by specific data source. Its key fields are:

- **diseaseId**: A string identifier for the associated disease.
- **targetId**: A string identifier for the associated target.
- **datatypeId**: The type of evidence data used (such as genetic association, somatic mutation, drug, etc.).
- **score**: A number quantifying the association strength, calculated from all direct evidence under that datatype, regardless of the datasource.
- **evidenceCount**: The count of evidence items supporting this association within this datatype group.

By aggregating scores across all sources for a given data type, this schema offers a broader but still very direct (no propagation) perspective—providing insight into which types of evidence most strongly link a disease and target, even if multiple sources are used for that data type.

**3. association_by_overall_direct**

This table aggregates all direct evidence, regardless of source or type, to provide a single overall association score between each disease and target pair. The fields include:

- **diseaseId**: The string identifier for the disease in the association.
- **targetId**: The unique string identifying the target.
- **score**: A numeric value representing the collective direct association strength, integrating all direct evidence and data types.
- **evidenceCount**: A numeric value indicating the total number of direct evidence items backing this overall association.

Unlike the previous tables, this schema delivers a holistic, direct view of the disease-target landscape—summarizing every type and source of direct evidence into an all-in-one score for each pair, without factoring in ontology-propagated relationships.

**4. association_by_datasource_indirect**

This table captures associations between diseases and targets, but, unlike the “direct” tables, it incorporates *ontology propagation*—meaning evidence from closely related diseases or targets (according to an ontology) can “propagate” to strengthen the association. The association score is calculated *for each combination* of data type and data source, such as “genetic association from ChEMBL,” but now considers not only primary data but also relevant evidence inferred from relationships within biomedical ontologies (for example, parent or child terms in disease or gene hierarchies). The key fields are:
- **datatypeId**: Type of data used (e.g., “genetic_association”).
- **datasourceId**: Identifier for the data source (e.g., “chembl”).
- **diseaseId**: Identifier for the disease.
- **targetId**: Identifier for the target (gene/protein).
- **score**: Numeric association value reflecting both direct and propagated evidence from this data source/type combo.
- **evidenceCount**: Number of evidence items (both direct and indirectly propagated) supporting this association.

This approach is especially useful for uncovering stronger or less obvious links, leveraging the knowledge encoded in biomedical ontologies.

**5. association_by_datatype_indirect**

This table, like its “direct” counterpart, groups association evidence by *data type* (not by individual datasource), but incorporates ontology propagation. It aggregates all direct and propagated pieces of evidence from all sources under a given data type (e.g., all “genetic_association” evidence across the resource, plus related propagated evidence), giving an overall strength score. The schema has:
- **diseaseId**: Disease identifier.
- **targetId**: Target identifier.
- **datatypeId**: Data type grouping.
- **score**: Association strength from all evidence of this type, factoring in ontology propagation.
- **evidenceCount**: Total number of pieces of supporting evidence (direct or inherited) under this grouping.

This provides a panoramic view of “how much evidence of type X (e.g., genetic) links target Y with disease Z,” with the benefit of added connections via ontology expansion.

**6. association_by_overall_indirect**

This is the most integrative association view. It collapses all data types and data sources and includes all *direct* and *propagated* evidence, yielding a single overall association score for each disease-target pair—regardless of origin or type of information. Its structure is:
- **diseaseId**: Identifier for the disease.
- **targetId**: Identifier for the target.
- **score**: The overall strength of association between this disease and target, *including* all direct and ontology-propagated evidence.
- **evidenceCount**: The grand total of pieces of supporting evidence, whether primary or indirect.

This schema is ideal for summarizing the totality of evidence—both explicitly reported and inferred—behind each disease-target association in the dataset.

**7. expression**

The `expression` table focuses on gene expression information across various tissues and biosamples. Each entry corresponds to an **Ensembl human gene** and includes an array of detailed tissue-specific data, allowing for deep exploration of where and how strongly a gene is expressed in the body. The schema contains:
- **id**: The Ensembl gene identifier for which the expression data is recorded.
- **tissues**: An array where each entry describes expression in a specific biosample. For each tissue, the following details are included:
  - **efo_code**: The ontology ID (from the Experimental Factor Ontology or Uberon) representing the tissue or biosample.
  - **label**: The human-readable name of the biosample.
  - **organs**: A list of organ names where the biosample is found.
  - **anatomical_systems**: A list of anatomical systems (e.g., nervous system, digestive tract) relevant to the sample.
  - **rna**: Contains objects with:
    - **value**: Numeric expression value.
    - **zscore**: The expression z-score.
    - **level**: RNA expression, normalized (e.g., on a 0-5 scale, or -1 if absent).
    - **unit**: Measurement unit (e.g., TPM, RPKM).
  - **protein**: Contains:
    - **reliability**: Placeholder for evaluating the quality of protein measurements.
    - **level**: Protein expression level on a normalized scale.
    - **cell_type**: An array, each entry describing the cell type assessed, with cell-type name, reliability, and expression level.

This schema enables researchers to pinpoint in which tissues or cell types—at both the RNA and protein level—a specific gene is active, providing essential context for understanding gene function and disease relevance.

**8. biosample**

The `biosample` table offers extensive metadata for each biological sample or tissue referenced elsewhere in the datasets. Each biosample entry includes:
- **biosampleId**: A unique ontology-based identifier.
- **biosampleName**: The human-readable name of the biosample.
- **description**: Additional narrative description about the sample.
- **xrefs**: A list of cross-reference IDs pointing to equivalent concepts in other biological ontologies.
- **synonyms**: List of synonymous names or labels for the biosample.
- **parents**: Direct parent IDs within the ontology, describing hierarchical relationships.
- **ancestors**: Extended ancestry in the biosample ontology hierarchy.
- **children** and **descendants**: IDs for direct children and extended offspring in the ontology.

By using ontology relationships and synonyms, this table underpins all analyses requiring context about anatomical source material, including mapping experimental data across studies, resolving sample ambiguities, and connecting with external data sources.

**9. colocalisation**

The `colocalisation` table documents results from analyses looking for shared genetic signals (typically across loci) that may drive both molecular and disease traits. Each entry represents a pair of study loci (such as from a GWAS for a disease and an eQTL for gene expression) where statistical evidence supports overlap. Key fields include:
- **leftStudyLocusId**, **rightStudyLocusId**: Identifiers for the two loci involved.
- **chromosome**: The chromosome where the colocalisation occurs.
- **rightStudyType**: The type of study for the right locus (e.g., GWAS, eQTL).
- **numberColocalisingVariants**: Count of overlapping genetic variants.
- **h0 – h4**: Posterior probabilities for different hypotheses about the association (e.g., h4 is probability both traits share a causal variant; h3 is both are associated but with different causal variants, etc.).
- **colocalisationMethod**: Analysis method name (such as coloc).
- **betaRatioSignAverage**: The average sign of the effect size (beta) ratio for the colocalising variants.

This schema is essential to bridge genetic evidence between molecular traits (like gene expression) and disease phenotypes, helping researchers pinpoint regions where the same genetic factors may underlie both.

**10. colocalisation_ecaviar**

This table presents the results of *colocalisation analysis* specifically performed with the eCAVIAR method—a statistical tool for determining whether two genetic signals (from different studies, such as a disease GWAS and an eQTL) are likely driven by the same underlying causal variant. Each record describes a colocalisation event:
- **leftStudyLocusId / rightStudyLocusId**: Identifiers for both sides of the colocalisation (the GWAS locus and the functional/molecular study locus).
- **chromosome**: The chromosome where the overlap occurs.
- **rightStudyType**: The type of the right-hand study (e.g., GWAS, eQTL).
- **numberColocalisingVariants**: The number of variants found at the intersection of the two loci.
- **clpp**: Colocalisation Posterior Probability (CLPP)—the probability assigned by eCAVIAR that both signals share the same causal variant.
- **colocalisationMethod**: Should report “eCAVIAR”.
- **betaRatioSignAverage**: The average sign of the beta ratio between the colocalised variants (captures directional consistency).

This schema enables integration of molecular and disease association evidence at a much finer, locus-specific level, as interpreted by eCAVIAR.

**11. credible_set**

The `credible_set` table details the fine-mapping of genetic association signals within a study region—identifying the most likely causal variants responsible for a genetic signal. Each entry includes:
- **studyLocusId**: Identifier for the locus and credible set.
- **studyId**: The parent study’s identifier.
- **variantId**: The leading variant (with highest posterior probability) for this credible set.
- **chromosome**, **position**: Genomic details.
- **region**: Genomic boundaries of the fine-mapping interval.
- **beta**, **zScore**, **pValueMantissa/Exponent**: Association statistics for the lead variant.
- **effectAlleleFrequencyFromSource**, **standardError**: Additional statistical measures.
- **qualityControls**: Array of QC flags.
- **finemappingMethod**: Method used for credible set derivation.
- **credibleSetIndex**: Indicates order when multiple credible sets exist for a locus.
- **credibleSetlog10BF**: Bayes factor for the credible set (measures weight of evidence).
- **purityMeanR2 / purityMinR2**: LD (linkage disequilibrium) metrics signifying how tightly variants are correlated within the set.
- **locusStart / locusEnd**: Start and end positions of the fine-mapped region.
- **sampleSize**: Study sample size that informed the mapping.
- **ldSet**: List of linked variants in LD with the lead variant, along with r².
- **locus**: Details of all variants in the credible set, with probability, Bayes Factor, association stats, and flags for inclusion in 95% or 99% credible set.
- **studyType**: Whether the credible set was derived from a GWAS or molecular QTL.
- **isTransQtl**: Whether this is a trans-QTL.

Fine-mapping through credible sets helps researchers zero in on the most promising candidate variants in genetic associations.

**12. disease**

This table contains ontology-based metadata for diseases referenced throughout the dataset. Each record describes a disease entry with:
- **id**: The Open Targets disease identifier.
- **code**: A URL/ref to a disease resource.
- **name**: Disease name.
- **description**: Narrative description.
- **dbXRefs**: Cross-references to other disease ontologies and databases.
- **parents / children / ancestors / descendants**: Relationships in the disease ontology hierarchy (supporting disease family, subtypes, etc).
- **synonyms**: Object grouping lists of exact, related, narrow, and broad synonyms.
- **obsoleteTerms / obsoleteXRefs**: Deprecated or merged terms.
- **therapeuticAreas**: Which “root” therapeutic area the disease belongs to.
- **ontology**: Further structure, indicating if this is a therapeutic area, leaf node, and sources for ontology mapping.

This is an essential lookup for all disease concept normalization, relationship resolving, and cross-referencing across biomedical resources.

**13. disease_phenotype**  
This table connects diseases to specific phenotypes (observable characteristics or traits), which are often used for clinical diagnosis or research.  
- **disease**: The identifier for the disease.
- **phenotype**: The identifier for the associated phenotype (could be HPO term, etc.).
- **evidence**: An array containing supporting evidence, where each entry details:
  - **aspect**: Type of biological or clinical information (e.g., genetic, clinical, etc.).
  - **bioCuration**: Whether the evidence is manually curated.
  - **diseaseFromSourceId/diseaseFromSource**: The ID and name of the disease in the original data source.
  - **diseaseName**: Standardized disease name.
  - **evidenceType**: The type/category of supporting evidence.
  - **frequency**: Observed frequency of the phenotype in patients with this disease.
  - **modifiers/onset/qualifier/qualifierNot/sex**: Modifiers and context of the phenotype association (e.g., age of onset, severity).
  - **references**: List of supporting literature or references (often PubMed IDs).
  - **resource**: The source database/resource for the evidence.

**14. drug_indication**  
This table covers approved and investigational indications for drugs—essentially, which diseases each drug is intended to treat.  
- **id**: Open Targets molecule (drug) identifier.
- **indications**: Array where each entry includes:
  - **disease**: List of disease identifiers this drug is indicated for.
  - **efoName**: Names of those diseases.
  - **references**: Source(s) of the indication (e.g., regulatory documents, clinical trial reports) with supporting IDs.
  - **maxPhaseForIndication**: The highest clinical trial phase reached for the indication.
- **approvedIndications**: List of disease identifiers for which the drug is approved.
- **indicationCount**: Total number of indications for this drug.

**15. drug_warning**  
This table logs major safety warnings associated with drugs, such as black box warnings or withdrawals.
- **chemblIds**: Open Targets molecule identifiers of drugs affected.
- **toxicityClass**: The type or class of safety issue (e.g., hepatotoxicity).
- **country**: The country where the warning was issued.
- **description**: Narrative description of the warning/adverse effect.
- **id**: Internal warning record ID.
- **references**: Array of supporting sources:
  - **ref_id**: Reference identifier (e.g., document, regulatory decision).
  - **ref_type**: Reporting body (FDA, EMA, etc.).
  - **ref_url**: URL to the external source.
- **warningType**: Type of regulatory action (withdrawn, black box warning, etc.).
- **year**: Year the warning was issued.
- **efo_term/efo_id**: Disease term(s)/identifier(s) relevant to the warning.
- **efo_id_for_warning_class**: Disease ID categorizing the warning’s underlying disease class.

**16. drug_molecule**
- Describes the properties, identifiers, and regulatory status of a drug molecule.
  - **id**: Open Targets molecule (drug) identifier.
  - **canonicalSmiles**: Standardized SMILES string for the molecule’s chemical structure.
  - **inchiKey**: InChIKey identifier (unique for molecular structure).
  - **drugType**: Type of drug (e.g., Antibody, Small molecule).
  - **blackBoxWarning**: Boolean flag—does the drug carry a black box safety warning?
  - **name**: Generic/primary name of the drug.
  - **yearOfFirstApproval**: Regulatory approval year (if any).
  - **maximumClinicalTrialPhase**: Highest achieved clinical trial phase.
  - **parentId**: Parent molecule identifier, if this drug is a derivative.
  - **hasBeenWithdrawn / isApproved**: Has the drug been withdrawn; is it officially approved?
  - **tradeNames**: List of brand/trade names for the molecule.
  - **synonyms**: Alternative names.
  - **crossReferences**: Array of identifiers and source database names for this molecule in other chemical and regulatory databases.
  - **childChemblIds**: (If relevant) IDs of derivative compounds.
  - **linkedDiseases / linkedTargets**: Diseases and targets the drug is associated with.
  - **description**: Clinical development summary.

**17. pharmacogenomics**
- Details genetic variants that affect drug response—relevant for personalized medicine.
  - **datasourceId, datasourceVersion**: Where the data comes from and its version.
  - **datatypeId**: Class of pharmacogenomic evidence (e.g., clinical_annotation).
  - **directionality**: Does the variant increase or decrease the drug response?
  - **evidenceLevel**: Strength of scientific evidence (e.g., clinical guideline, case report).
  - **genotype, genotypeAnnotationText, genotypeId**: Genetics of the variant (and explanation).
  - **haplotypeFromSourceId, haplotypeId**: Haplotype information.
  - **literature**: List of supporting PubMed IDs.
  - **pgxCategory**: Category of impact (e.g. Toxicity).
  - **phenotypeFromSourceId, phenotypeText**: Disease or effect caused by the variant.
  - **variantAnnotation**: Placeholder for further variant annotation details.
  - **studyId, targetFromSourceId**: Associated study and target.
  - **variantFunctionalConsequenceId**: Ontology ID for functional consequence.
  - **variantRsId, variantId**: dbSNP and internal variant IDs.
  - **isDirectTarget**: Is this variant in a direct target of the drug?
  - **drugs**: A list of other drugs impacted by the same genetic variation.

**18. study**
- Describes GWAS, molecular QTL, or other -omics/clinical studies referenced in the dataset.
  - **studyId**: Unique identifier for the study (GWAS/QTL/etc.).
  - **geneId**: Ensembl gene ID (for QTLs).
  - **projectId**: Source project identifier.
  - **studyType**: e.g., “gwas” or “molQTL” or “clinical”.
  - **traitFromSource**: Raw trait/disease/phenotype label from the study.
  - **traitFromSourceMappedIds**: Standardized trait/disease identifier mapping(s).
  - **biosampleFromSourceId**: Biosample identifier linked to the study.
  - **pubmedId**: PubMed reference if study is published.
  - **publicationTitle, publicationFirstAuthor, publicationDate, publicationJournal**: Bibliographic info.
  - **backgroundTraitFromSourceMappedIds**: Any shared background traits in the study.
  - **initialSampleSize**: Initial number of samples included.
  - **nCases, nControls, nSamples**: Case/control/sample counts for GWAS/meta-analysis.
  - **cohorts**: Cohort names or identifiers.
  - **ldPopulationStructure**: Ancestry/population structure details.
  - **discoverySamples / replicationSamples**: Array of ancestry/sample details for discovery vs. replication phase.
  - **qualityControls, analysisFlags**: List of quality control and analysis flags.
  - **summarystatsLocation, hasSumstats**: File location and availability of summary stats.
  - **condition**: Medical or experimental condition.
  - **sumstatQCValues**: Additional QC on summary statistics.
  - **diseaseIds / backgroundDiseaseIds**: Disease(s) associated with or background to the study.
  - **biosampleId**: Reference to a biosample from the ontology, if present.

**19. known_drug**
- Captures the relationships between drugs, their molecular targets, and the diseases they are being used or tested to treat.
  - **drugId**: Open Targets molecule (drug) identifier.
  - **targetId**: Open Targets identifier for the drug’s molecular target (e.g., a gene).
  - **diseaseId**: Open Targets identifier for the disease/condition being treated.
  - **phase**: Highest clinical trial phase reached for this drug-target-disease combination.
  - **status**: Clinical trial or approval status (e.g. “Approved”, “Withdrawn”, “In development”).
  - **urls**: List of supporting web URLs (e.g., clinical trials, regulatory filings), each with a human-readable name and a target URL.
  - **ancestors**: List of parent disease terms (ontology).
  - **label**: Human-readable disease name for the indication.
  - **approvedSymbol/approvedName**: Symbols/names for the target modulated by the drug.
  - **targetClass**: Biological classification(s) for the target (e.g. “Enzyme”).
  - **prefName/tradeNames/synonyms**: Common, trade, and alternative names for the drug.
  - **drugType**: Drug modality/type (small molecule, antibody, etc.).
  - **mechanismOfAction**: How the drug affects its biological target.
  - **targetName**: Full name of the gene/protein targeted by the drug.

**20. literature**
- Contains bibliographic information for literature and publications referenced in other parts of the schema.
  - **pmid**: PubMed identifier.
  - **pmcid**: PubMed Central (PMC) identifier.
  - **date/year/month/day**: Publication date.
  - **keywordId**: Unique identifier for keywords indexed in this paper.
  - **relevance**: Numeric relevance score for keywords in context.
  - **keywordType**: Classification of the keyword: disease/syndrome (DS), gene/protein (GP), chemical/drug (CD).

**21. literature_vector**
- Stores extracted, normalized, and embedded representations of key biomedical terms from the literature, for semantic and ML modeling.
  - **category**: What kind of entity the term is (target, drug, or disease).
  - **word**: The normalized form of the entity.
  - **norm**: Numeric normalization value (or magnitude).
  - **vector**: An array representing the word in a high-dimensional semantic vector space (word embeddings), used for similarity or machine learning tasks.

**22. l2g_prediction**
- Contains results of "locus-to-gene" (L2G) machine-learning predictions that estimate which gene is likely to be causal at a GWAS or molecular QTL locus.
  - **studyLocusId**: Unique identifier for the locus (from studies or credible sets).
  - **geneId**: Ensembl gene ID of the plausible causal gene.
  - **score**: L2G prediction score for the locus-gene pair (higher = greater confidence the gene is causal at this locus).
  - **features**: Array with details on model features contributing to the score, each with a name, value, and SHAP values (explaining model attribution).
  - **shapBaseValue**: The SHAP model’s baseline value for reference.

**23. interaction**
- Describes binary molecular interactions, such as protein-protein, protein-DNA, or protein-small molecule interactions, supported by experimental or curated evidence.
  - **sourceDatabase**: Which database reported the interaction.
  - **targetA/targetB**: Open Targets IDs for the two interacting molecules (typically Ensembl Gene IDs).
  - **intA/intB**: Source identifiers for the interactors.
  - **intABiologicalRole/intBBiologicalRole**: Biological roles of each interactor in the interaction.
  - **speciesA/speciesB**: Taxonomic info for A and B (with mnemonic, scientific name, taxon ID).
  - **count**: Number of supporting pieces of evidence.
  - **scoring**: Confidence or interaction score.

**24. interaction_evidence**
- Detailed evidence for a specific molecular interaction, supporting experimental context and provenance.
  - **participantDetectionMethodA/B**: Arrays of detection methods (with MI identifiers and short names) used for each participant.
  - **hostOrganismTaxId**: NCBI taxonomy ID of the host organism in which the interaction was observed.
  - **targetB**: Open Targets ID of the second interaction partner.
  - **evidenceScore**: Numerical value indicating strength/confidence of the evidence.
  - **expansionMethodShortName/expansionMethodMiIdentifier**: Description/identifier of method used to expand the interaction dataset.
  - **hostOrganismScientificName**: Scientific name of the host organism.
  - **intBBiologicalRole**: Role of targetB in the context of the interaction.
  - **interactionResources**: Object with version, source database, and detection method details.
  - **interactionTypeShortName/interactionTypeMiIdentifier**: Names/identifiers for the type of interaction.
  - **interactionIdentifier**: Unique identifier linking to this specific evidence at the source.
  - **hostOrganismTissue**: Detailed tissue info (name, cross-references) of the host organism where experiment was performed.

**25. mouse_phenotype**
- Contains information on mouse/animal models used to assess phenotypic consequences of gene perturbations (e.g., knockout), linking genes/targets to clinically relevant traits.
  - **biologicalModels**: Array of detailed mouse models, each with:
    - **allelicComposition**: Genetic makeup of the mouse model.
    - **geneticBackground**: Strain/background on which the model is built.
    - **id**: Unique ID for the model (MGI).
    - **literature**: Array of PubMed references.
    - **modelPhenotypeClasses**: Array of phenotype classes/ontologies (ID & label).
    - **modelPhenotypeId/modelPhenotypeLabel**: Detailed observed phenotype(s) in the model.
    - **targetFromSourceId**: Identifier for the human gene being studied.
    - **targetInModel**: Gene name in the mouse model.
    - **targetInModelEnsemblId/targetInModelMgiId**: Mouse gene identifiers (Ensembl/MGI).

**26. openfda_significant_adverse_drug_reactions**
- Reports statistically significant adverse drug reactions from the US FDA’s OpenFDA database.
  - **chembl_id**: Drug or clinical candidate identifier (ChEMBL).
  - **event**: Reported adverse event (e.g., side effect name).
  - **count**: Number of reported cases.
  - **llr**: Log-likelihood ratio quantifying strength of the association.
  - **critval**: Statistical critical value for significance.
  - **meddraCode**: MedDRA ontology code for the adverse event.

**27. openfda_significant_adverse_target_reactions**
- Reports significant adverse events observed across all drugs that act on a particular molecular target, using FDA pharmacovigilance data.
  - **targetId**: Unique identifier for the gene/protein target.
  - **event**: Name of the adverse reaction.
  - **count**: Number of reported instances (summed over all relevant drugs).
  - **llr**: Log-likelihood ratio (strength of link between target and event).
  - **critval**: Statistical threshold for significance.
  - **meddraCode**: Ontology code for the adverse event (MedDRA).

**28. so**
- This entity defines a Sequence Ontology (SO) term, which is used to describe sequence-based features or consequences in genomic data.
  - **id**: Unique identifier for the SO term (e.g., SO:0001583).
  - **label**: Human-readable name for the sequence ontology term.

**29. target**
- Provides detailed information about a molecular target, often a gene or protein, particularly in the context of drug discovery or disease association.
  - **id**: Unique identifier for the target (typically Ensembl gene ID).
  - **approvedSymbol**: The standardized gene symbol approved by HGNC.
  - **biotype**: Category of the gene (e.g., protein_coding).
  - **transcriptIds**: List of all Ensembl transcript IDs for the target gene.
  - **canonicalTranscript**: Details of the canonical (main) transcript of the gene, including Ensembl transcript ID, chromosome, start, end, and strand.
  - **canonicalExons**: List of exons for the canonical transcript.
  - **genomicLocation**: Chromosome, start, end, and strand info for the gene's genomic location.
  - **alternativeGenes**: Alternative Ensembl gene IDs from non-canonical chromosomes.
  - **approvedName**: Full descriptive name of the gene.
  - **go**: List of Gene Ontology (GO) annotations, each with GO ID, source, evidence, aspect (Function, Process, or Compartment), gene product, and ECO ID.
  - **hallmarks**: Cancer attributes linked to the target (from COSMIC), with supporting PubMed IDs and attribute info.
  - **cancerHallmarks**: Specific cancer hallmark associations, if any, similar in structure.

**30. synonyms, symbolSynonyms, nameSynonyms**
- These three arrays (as properties of the target entity) give alternative names for the target gene:
  - **synonyms**: List of objects, each with a synonym label and its source, including both symbol and name synonyms.
  - **symbolSynonyms**: Alternate gene symbols for the target, with sources.
  - **nameSynonyms**: Alternative names (not symbols) for the gene, with sources.

**31. functionDescriptions**
- An array of brief textual descriptions of the biological role or function of a target gene or protein.
  - **Each entry**: A string summarizing a known function, often sourced from UniProt or other protein/gene databases.

**32. subcellularLocations**
- Describes where the gene product (usually a protein) is found inside the cell.
  - **location**: Name of the subcellular compartment (e.g., "nucleus", "plasma membrane").
  - **source**: Source database for this location info.
  - **termSL**: Unique term identifier from SwissProt for the subcellular location.
  - **labelSL**: High-level label/category of the subcellular location from SwissProt.

**33. targetClass**
- Array defining the classification(s) of the target gene or protein, usually from ChEMBL.
  - **id**: Unique identifier for the class.
  - **label**: Human-friendly class label (e.g., "Kinase", "GPCR", etc.).
  - **level**: Describes where in the classification hierarchy this class sits (e.g., family, subgroup).

**34. obsoleteSymbols**  
- An array listing old or deprecated symbols (short gene names) that were previously used for the gene/target, along with where that symbol came from.
  - **label**: The obsolete gene symbol.
  - **source**: The database or resource that used that old symbol.

**35. obsoleteNames**  
- An array listing outdated full names for the gene/target, each paired with the source where it was used.
  - **label**: The obsolete full name.
  - **source**: The origin database/resource for that obsolete name.

**36. constraint**  
- An array containing information about the genetic constraint of the target gene, generally indicating how intolerant it is to genetic variation (important for interpreting mutation impacts). Data is typically from GnomAD.
  - **constraintType**: The type of constraint being measured (e.g., "pLI score", "LOF constraint").
  - **score**: The constraint score quantifying intolerance.
  - **exp**: The expected value for this constraint.
  - **obs**: The observed value for this constraint.
  - **oe**: Observed/Expected ratio.
  - **oeLower/oeUpper**: Lower and upper confidence bounds for the observed/expected ratio.
  - **upperRank**: Rank of the gene among all tested concerning constraint.
  - **upperBin**: Bin/category for the score (for quick lookups or summary assessments).
  - **upperBin6**: Category using a six-bin scale (for additional stratification).

**37. tep**
- This is an object with information related to a "Target Enabling Package" (TEP) for the target gene.
  - **targetFromSourceId**: The Ensembl gene ID for the TEP target.
  - **description**: Description of what the TEP provides or covers for this target.
  - **therapeuticArea**: The medical or disease area that the TEP is relevant for.
  - **url**: A web link to more information about the TEP for this target.

**38. proteinIds**
- An array of objects listing protein identifiers associated with the target gene, from various reference databases.
  - **id**: The protein ID (e.g., UniProt, RefSeq, etc.).
  - **source**: The database where this protein ID comes from.

**39. dbXrefs**
- An array of objects giving database cross-references for the target gene.
  - **id**: Cross-reference ID for the external database.
  - **source**: Name of the database the cross-reference points to.


**40. chemicalProbes**
- An array of objects describing chemical probes linked to the target gene for experimental purposes.
  - **control**: Indicator of whether the probe is a control compound.
  - **drugId**: Drug ID associated with the probe.
  - **id**: Unique identifier for the probe.
  - **isHighQuality**: Boolean indicating high quality as a probe.
  - **mechanismOfAction**: Array listing mechanisms by which the probe acts.
  - **origin**: Array of origins for the probe.
  - **probeMinerScore**: Score from ProbeMiner for probe quality.
  - **probesDrugsScore**: Score related to druggability.
  - **scoreInCells**: Activity score in cells.
  - **scoreInOrganisms**: Activity score in organisms.
  - **targetFromSourceId**: Ensembl gene ID of the target for the probe.
  - **urls**: Array of objects with `niceName` and `url` to reference more info about the probe.

**41. homologues**
- An array of objects describing homologous genes in other species.
  - **speciesId**: Species identifier.
  - **speciesName**: Name of the species.
  - **homologyType**: Type of homology (e.g., ortholog, paralog).
  - **targetGeneId**: ID of the homologous gene.
  - **isHighConfidence**: (string or boolean) High-confidence match or not.
  - **targetGeneSymbol**: Gene symbol of the homologue.
  - **queryPercentageIdentity**: % identity in query gene.
  - **targetPercentageIdentity**: % identity in homologue.
  - **priority**: Priority score for the homologue.

**42. tractability**
- Array of tractability information about the target’s potential as a drug target.
  - **modality**: Type of modality (e.g., small molecule, antibody).
  - **id**: Unique tractability category ID.
  - **value**: Boolean indicating if the target is tractable under this modality.

**43. safetyLiabilities**
- An array of objects with safety-related information for the target.
  - **event**: Name of the safety event associated with the target (e.g., toxicity).
  - **eventId**: Unique identifier for the safety event.
  - **effects**: Array of objects describing reported effects for the safety event with fields such as:
    - **direction**: Direction of the effect (e.g., increase, decrease).
    - **dosing**: Information about dosing related to the effect.
  - **biosamples**: Array of objects describing biosamples used in the safety assessment with fields such as:
    - **cellFormat**: Format of the biosample cells.
    - **cellLabel**: Name or label of the biosample cell.
    - **tissueId**: Tissue ontology ID for the biosample.
    - **tissueLabel**: Name of the tissue.
  - **datasource**: Name of the data source reporting the safety liability.
  - **literature**: Literature references supporting the safety event.
  - **url**: URL to more info about the safety liabilities.
  - **studies**: Array of objects for studies linked to the safety assessment (description, name, type).

**44. pathways**
- An array of objects describing pathway annotations for the target.
  - **pathwayId**: Unique identifier for the pathway.
  - **pathway**: Name of the pathway.
  - **topLevelTerm**: Top-level category or label for the pathway (for classification).

**45. tss**
- A numeric field indicating Transcription Start Site (TSS) information for the target gene or transcript.

**46. gene_essentiality**
- An object describing gene essentiality measurements from CRISPR screening data.
  - **id**: Ensembl target identifier [bioregistry:ensembl].
  - **geneEssentiality**: Array of objects, each with:
    - **depMapEssentiality**: Array of objects for DepMap data, including:
      - **screens**: List of CRISPR screen experiments (with cell line, effect, mutation, etc.).
      - **tissueId/tissueName**: IDs and names for the tissue.
      - **isEssential**: Boolean indicating essentiality in context.

**47. evidence**
- Array of evidence records supporting associations/clinical claims. Contains several context-rich attributes, such as:
  - **datasourceId**: Identifier for source.
  - **targetId**: Open Targets target identifier.
  - **alleleOrigins**, **allelicRequirements**, **ancestry**: Genetics context.
  - **statisticalMethod**, **statisticalMethodOverview**, **studyId**, etc.

**48. biomarkers**
- An object containing information about biomarkers for the target.
  - **geneExpression**: Array of gene expression altering biomarkers (with ID, name, etc.).
  - **geneticVariation**: Array of genetic variation biomarkers (with variant info, functional consequences, etc.).
