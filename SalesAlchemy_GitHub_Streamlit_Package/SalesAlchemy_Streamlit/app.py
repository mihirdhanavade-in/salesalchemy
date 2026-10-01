import streamlit as st

st.set_page_config(page_title='SalesAlchemy', page_icon='⚗️', layout='wide')

import hashlib
import json
import pandas as pd
import os
import engine

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {font-family:'Inter',sans-serif;color:#0F172A;}
h1,h2,h3,h4 {font-family:'Plus Jakarta Sans',sans-serif;}
.stApp {background:#F8FAFC;}
.block-container {padding-top:2rem;padding-bottom:3rem;}
#MainMenu,footer {visibility:hidden;}
.hero {border:1px solid #E2E8F0;background:linear-gradient(120deg,#FFFFFF,#EEF2FF,#F5F3FF);border-radius:24px;padding:30px;margin-bottom:20px;box-shadow:0 10px 30px rgba(15,23,42,.05);}
.wordmark {display:flex;align-items:center;gap:12px;font-family:'Plus Jakarta Sans',sans-serif;font-size:38px;font-weight:800;}
.alchemy {background:linear-gradient(90deg,#4F46E5,#7C3AED);-webkit-background-clip:text;background-clip:text;color:transparent;}
.tagline {margin-top:12px;color:#475569;font-size:17px;}
.badge {display:inline-block;margin-top:18px;padding:5px 12px;border-radius:20px;background:#FEF3C7;color:#78350F;font-size:12px;font-weight:600;}
[data-testid="stForm"],[data-testid="stExpander"] {background:white;border:1px solid #E2E8F0;border-radius:16px;}
button[kind="primary"] {background:linear-gradient(90deg,#4F46E5,#7C3AED);border:none;border-radius:10px;color:white;}
[data-testid="stMetric"] {background:white;padding:18px;border:1px solid #E2E8F0;border-radius:14px;}
</style>
"""
HERO = """
<div class="hero"><div class="wordmark">
<svg width="48" height="48" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" aria-label="Flask and spark">
<path d="M17 5h14M20 5v14L10 36q-3 7 5 7h18q8 0 5-7L28 19V5" fill="none" stroke="#4F46E5" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M16 31h16l5 8H11z" fill="#7C3AED" opacity=".35"/>
<path d="M38 7l2 5 5 2-5 2-2 5-2-5-5-2 5-2z" fill="#F5B301"/>
<circle cx="21" cy="33" r="2" fill="#F5B301"/></svg>
<span>Sales<span class="alchemy">Alchemy</span></span></div>
<div class="tagline">Turn customer requirements into winning proposals.</div>
<div class="badge">TEC 406 · Project 8 · Sales &amp; CRM</div></div>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)
st.warning('AI-generated draft - review before sending.')

def setting(name, default=''):
    try:
        return str(st.secrets.get(name, os.environ.get(name, default)))
    except FileNotFoundError:
        return os.environ.get(name, default)


HF_MODEL_ID = setting('HF_MODEL_ID')
LIVE_ENABLED = bool(setting('HF_TOKEN') and HF_MODEL_ID)


def load_llm():
    return engine.ModelManager(setting('HF_TOKEN'))


def get_retriever(documents, demo):
    signature = hashlib.sha256((json.dumps(documents, sort_keys=True, ensure_ascii=False) + str(demo)).encode('utf-8')).hexdigest()
    if st.session_state.get('retriever_signature') != signature:
        st.session_state.pop('retriever', None)
        st.session_state['retriever'] = engine.Retriever(documents, use_embeddings=False)
        st.session_state['retriever_signature'] = signature
    return st.session_state['retriever']

def show_score(bundle):
    result = bundle['evaluation']
    st.metric('Prompt-quality score', str(result['total']) + ' / 100')
    st.dataframe(pd.DataFrame(result['metrics']), hide_index=True)
    if result['mismatches']:
        st.error('Currency figures not present in the calculated table: ' + ', '.join(result['mismatches']))
    if result['risk']:
        st.warning('Potential risky claims: ' + ', '.join(sorted(set(result['risk']))))
    if result['placeholders']:
        st.warning('Unfilled placeholders: ' + ', '.join(result['placeholders']))

def show_sources(bundle):
    with st.expander('Sources used'):
        st.caption('These are the exact retrieved chunks supplied to the prompts. Retrieval does not prove that every generated claim is supported.')
        for index, source in enumerate(bundle['sources'], start=1):
            st.markdown('**Source ' + str(index) + ': ' + source['source'] + '**')
            st.caption('Chunk ' + str(source['chunk']) + ' · Similarity ' + str(round(source['score'], 3)))
            st.text(source['text'])

with st.sidebar:
    st.header('Workspace')
    demo = st.toggle('Demo mode (no model)', value=True, key='demo_mode', disabled=not LIVE_ENABLED)
    if not LIVE_ENABLED:
        st.info('Demo is ready. To enable live AI, the app owner must add HF_TOKEN and HF_MODEL_ID in Streamlit Secrets.')
    st.caption('Model: ' + (HF_MODEL_ID or 'Demo — no model'))
    if not demo:
        st.info('Live generation sends the customer brief, retrieved company facts and pricing to Hugging Face and its inference provider. Provider charges may apply.')
    temperature = st.slider('Temperature', 0.1, 1.0, 0.7, 0.1, key='temperature')
    token_cap = st.slider('Maximum tokens per engineered section', 250, 600, 600, 50, key='token_cap')
    st.caption('Default temperature: 0.7. Individual sections also have their own smaller token budgets.')
    st.divider()
    st.subheader('Knowledge base')
    replace_sample = st.checkbox('Replace the fictional sample KB', value=False, key='replace_sample')
    uploaded_files = st.file_uploader('Upload company documents', type=['md','txt','json'], accept_multiple_files=True, key='kb_uploads')
    st.caption('UTF-8 files; maximum 200 KB each, 1 MB combined. Name a replacement rate card services_and_rates.json.')
    if st.button('Clear session data', key='clear_session'):
        st.session_state.clear()
        st.rerun()
    st.divider()
    st.caption('No permanent personal-data database. Inputs, uploads and drafts are held in the active session.')
    st.caption('Use fictional or anonymised information. The app owner controls viewer access in Streamlit settings.')

try:
    documents, rates = engine.load_knowledge([(file.name, file.getvalue()) for file in uploaded_files], replace_sample=replace_sample)
except Exception as error:
    st.error('Knowledge-base problem: ' + str(error))
    st.info('Uploading only a .txt or .md file? Untick Replace the fictional sample KB. Full replacement also needs services_and_rates.json.')
    st.stop()

rate_signature = hashlib.sha256(json.dumps(rates, sort_keys=True).encode('utf-8')).hexdigest()[:12]
with st.sidebar:
    currency = st.selectbox('Proposal currency', engine.CURRENCIES, index=engine.CURRENCIES.index(rates['currency']), key='currency_' + rate_signature)
    st.caption('Rate-card currency: ' + rates['currency'])
    if currency != rates['currency']:
        st.warning('Upload prices in the selected currency. Automatic currency conversion is not provided.')
    st.caption('Review the selected model licence and provider terms before commercial use.')

generate_tab, output_tab, evaluate_tab, about_tab = st.tabs(['Generate','Proposal Output','Evaluate','About & Governance'])
with generate_tab:
    st.subheader('Customer brief')
    if not replace_sample:
        st.info('FICTIONAL SAMPLE knowledge base is active. Uploaded documents supplement it, except a replacement rate card.')
    with st.form('proposal_form'):
        left, right = st.columns(2)
        with left:
            client = st.text_input('Client name', 'Example Manufacturing', max_chars=100, key='client')
            industry = st.text_input('Industry', 'Manufacturing', max_chars=100, key='industry')
            company_size = st.selectbox('Company size', ['1–10 employees','11–50 employees','51–200 employees','201–1000 employees','More than 1000 employees'], index=2, key='company_size')
            budget_min = st.number_input('Budget minimum', min_value=0.0, max_value=1_000_000_000.0, value=50000.0, step=5000.0, key='budget_min')
            budget_max = st.number_input('Budget maximum', min_value=0.0, max_value=1_000_000_000.0, value=150000.0, step=5000.0, key='budget_max')
        with right:
            timeline = st.text_input('Requested timeline', '6–8 weeks', max_chars=150, key='timeline')
            tone = st.selectbox('Tone', ['Professional','Consultative','Warm','Concise'], key='tone')
            discount = st.number_input('Discount %', 0.0, 100.0, 0.0, 0.5, key='discount')
            tax = st.number_input('Tax %', 0.0, 100.0, 18.0, 0.5, key='tax')
            st.caption('Tax is applied after discount. Confirm applicable tax treatment yourself.')
        requirements = st.text_area('Customer requirements', 'Track sales enquiries in a shared CRM pipeline, assign owners, create a sales dashboard, and train users.', height=130, max_chars=4000, key='requirements')
        st.markdown('**Services of interest and quantities**')
        quantities = {}
        for service in rates['services']:
            quantities[service['id']] = st.number_input(
                service['name'] + ' — ' + engine.fmt(service['unit_price'], rates['currency']) + ' / ' + service['unit'],
                min_value=0, max_value=10000, value=1 if service['id'] in {'discovery','crm'} else 0, step=1,
                key='quantity_' + rate_signature + '_' + service['id'], help='Planning timeline: ' + service['timeline'])
        submitted = st.form_submit_button('Generate proposal', type='primary')
    if submitted:
        try:
            if not client.strip() or not industry.strip():
                raise ValueError('Enter a client name and industry.')
            if len(requirements.strip()) < 20:
                raise ValueError('Enter at least 20 characters of requirements.')
            if budget_max < budget_min:
                raise ValueError('Budget maximum must be at least the minimum.')
            if not timeline.strip():
                raise ValueError('Enter the requested timeline.')
            price = engine.pricing(rates, quantities, discount, tax, currency)
            selected = [s['name'] for s in rates['services'] if quantities[s['id']] > 0]
            request = {'client':client.strip(), 'industry':industry.strip(), 'company_size':company_size,
                       'requirements':requirements.strip(), 'budget_range':engine.fmt(budget_min,currency)+' to '+engine.fmt(budget_max,currency),
                       'timeline':timeline.strip(), 'services':selected, 'tone':tone, 'currency':currency,
                       'discount_percent':discount, 'tax_percent':tax}
            settings = {'demo':demo, 'model_id':HF_MODEL_ID or 'Demo', 'temperature':temperature, 'token_cap':token_cap}
            with st.spinner('Preparing company sources'):
                retriever = get_retriever(documents, demo)
                query = requirements + ' ' + industry + ' ' + ' '.join(selected) + ' company approach payment terms validity exclusions'
                sources = retriever.search(query, k=4)
            st.caption('Retrieval: ' + retriever.mode)
            if retriever.warning:
                st.warning(retriever.warning)
            progress_bar = st.progress(0.0)
            def report(name, index, total):
                progress_bar.progress((index-1)/total, text='Working on ' + name)
            bundle = engine.run_engineered(request, price, sources, settings, manager=None if demo else load_llm(), progress=report)
            progress_bar.progress(1.0, text='Draft ready')
            st.session_state['bundle'] = bundle
            st.session_state.pop('comparison', None)
            total = float(price['grand_total'])
            if total > budget_max:
                st.warning('The calculated total exceeds the entered maximum budget.')
            elif total < budget_min:
                st.info('The calculated total is below the entered minimum budget.')
            st.success('Open Proposal Output to review your draft.')
        except Exception as error:
            st.error('Generation could not finish: ' + str(error) + ' Previous saved results, if any, remain available.')

with output_tab:
    bundle = st.session_state.get('bundle')
    if not bundle:
        st.info('Generate a proposal to see the five sections.')
    else:
        st.caption(bundle['mode']+' · '+bundle['settings']['model_id']+' · Saved input: '+bundle['request']['client'])
        with st.expander('Requirement analysis'):
            st.markdown(bundle['analysis'])
        for name in engine.SECTIONS:
            st.subheader(name)
            st.markdown(bundle['sections'][name])
            if name == 'Pricing Assumptions':
                st.markdown(bundle['pricing']['markdown'])
        show_sources(bundle)
        with st.expander('Saved input and generation settings'):
            st.json(bundle['request'])
            st.json(bundle['settings'])
        if bundle['evaluation']['mismatches']:
            st.error('The scorecard found currency mismatches. Resolve them before treating this as a commercial proposal.')
        reviewed = st.checkbox('I have reviewed this draft.', key='review_' + bundle['id'])
        export_error = None
        try:
            markdown_data = engine.export_markdown(bundle)
            docx_data = engine.export_docx(bundle)
        except Exception as error:
            export_error, markdown_data, docx_data = str(error), '', b''
        if export_error:
            st.error('Export problem: ' + export_error)
        left, right = st.columns(2)
        with left:
            st.download_button('Download DOCX', data=docx_data, file_name='SalesAlchemy_Proposal.docx',
                mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                disabled=not reviewed or bool(export_error), key='docx_' + bundle['id'])
        with right:
            st.download_button('Download Markdown', data=markdown_data, file_name='SalesAlchemy_Proposal.md',
                mime='text/markdown', disabled=not reviewed or bool(export_error), key='markdown_' + bundle['id'])

with evaluate_tab:
    bundle = st.session_state.get('bundle')
    if not bundle:
        st.info('Generate a proposal before evaluating it.')
    else:
        st.subheader('Current proposal scorecard')
        show_score(bundle)
        st.caption('This is a heuristic screening score, not proof of factual accuracy. Currency checking recognises the supported currency codes and symbols with ordinary numeric amounts. It does not understand amounts written in words, lakh/crore notation, or whether a valid figure has the correct semantic label.')
        st.subheader('A/B prompt comparison')
        st.write('Run a single-prompt baseline and a fresh engineered chain using the same saved requirements, prices, retrieved sources, model and sampling settings.')
        st.caption('Baseline gets the sum of the five output token budgets. The engineered method also uses a separate analysis call. Sampling is stochastic, so repeat live trials before drawing conclusions. The original proposal remains available.')
        if st.button('Run BASELINE vs ENGINEERED', key='run_ab'):
            try:
                settings = bundle['settings']
                manager = None if settings['demo'] else load_llm()
                bar = st.progress(0.0, text='Generating baseline')
                baseline = engine.run_baseline(bundle['request'],bundle['pricing'],bundle['sources'],settings,manager)
                bar.progress(1/7, text='Baseline complete')
                def ab_progress(name, index, total):
                    bar.progress(index/7, text='Engineered: ' + name)
                engineered = engine.run_engineered(bundle['request'],bundle['pricing'],bundle['sources'],settings,manager,ab_progress)
                st.session_state['comparison'] = {'Baseline':baseline, 'Engineered':engineered}
                bar.progress(1.0, text='Comparison complete')
            except Exception as error:
                st.error('A/B comparison failed: ' + str(error))
        comparison = st.session_state.get('comparison')
        rating_versions = {'Current engineered draft':bundle}
        if comparison:
            rating_versions = comparison
            table_rows = []
            for version, result in comparison.items():
                record = {'Version':version}
                record.update({row['Metric']:row['Score'] for row in result['evaluation']['metrics']})
                record['Total'] = result['evaluation']['total']
                table_rows.append(record)
            frame = pd.DataFrame(table_rows).set_index('Version')
            st.dataframe(frame)
            st.bar_chart(frame[['Total']])
            if bundle['settings']['demo']:
                st.warning('Illustrative demo comparison: both drafts are canned. These scores are not evidence of model improvement.')
            for version, result in comparison.items():
                with st.expander(version + ' draft and detailed score'):
                    show_score(result)
                    for name in engine.SECTIONS:
                        st.markdown('### ' + name)
                        st.markdown(result['sections'].get(name) or '(Section missing)')
                    if 'raw' in result:
                        st.markdown('**Unparsed baseline output**')
                        st.text(result['raw'])
        st.subheader('Human review')
        st.caption('Rate each displayed draft from 1 (poor) to 5 (excellent). Assess accuracy against the supplied facts and pricing.')
        records = []
        for version, result in rating_versions.items():
            st.markdown('**' + version + '**')
            columns = st.columns(4)
            record = {'version':version,'mode':result['mode'],'model':result['settings']['model_id'],'automated_score':result['evaluation']['total']}
            for column, metric in zip(columns,['clarity','relevance','persuasiveness','accuracy']):
                with column:
                    record[metric] = st.slider(metric.title(),1,5,3,key='rating_'+result['id']+'_'+metric)
            records.append(record)
        rating_identity = '_'.join(result['id'] for result in rating_versions.values())
        ratings_reviewed = st.checkbox('I have reviewed this draft.',key='ratings_review_'+rating_identity,help='For an A/B comparison, confirm review of both drafts.')
        st.download_button('Download human ratings CSV',data=engine.ratings_csv(records),file_name='SalesAlchemy_Human_Ratings.csv',mime='text/csv',disabled=not ratings_reviewed,key='ratings_download_'+rating_identity)

with about_tab:
    st.subheader('AI solution architecture')
    st.text('Customer brief + company documents\n  ↓\nValidation + deterministic Python pricing\n  ↓\nChunking → local TF-IDF → top-four source chunks\n  ↓\nHugging Face hosted inference → analysis + five sections\n  ↓\nScorecard + human review → DOCX / Markdown\n\nBaseline branch: same inputs → one generation → A/B scorecard')
    st.subheader('Prompt workflow')
    st.write('Step 0 analyses goals, pain points, constraints and decision criteria. Steps 1–5 write each output section separately. This cloud edition uses lightweight TF-IDF retrieval. Every engineered prompt receives structured requirements, retrieved company facts, authoritative pricing, output rules and restrictions on unsupported claims.')
    st.subheader('Privacy and governance')
    st.markdown('''
- **No permanent personal-data database:** application inputs, uploads, drafts and ratings are held in session memory. The app does not deliberately write them into application files or logs. Downloaded files are saved by the user. Fictional sample documents are stored with the app. No model weights are downloaded by this cloud edition.
- **Session scope:** closing a tab does not promise immediate memory erasure. Clear session data or stop the runtime when done.
- **Hosting and live inference:** viewer access is controlled in Streamlit settings. Live generation sends requirements, retrieved text and pricing to Hugging Face and its selected provider; their processing and retention policies apply. Use anonymised information.
- **Hallucination warning:** retrieved facts and a high score do not establish factual correctness. Review every claim.
- **Prompt injection:** documents are treated as untrusted data. Prompt instructions reduce risk but are not a security boundary.
- **Model licences and costs:** the app owner selects a supported hosted chat model. Review its licence and provider terms. Inference may consume credits or incur usage charges. Demo mode makes no inference requests.
- **Bias and fairness:** avoid stereotypes, discriminatory assumptions, manipulative urgency and unsupported claims. Review tone for the intended audience.
- **Human control:** no email is sent automatically. Downloads require a review checkbox.
''')
    st.subheader('Business impact')
    st.write('Illustrative assumption: 20 proposals per month, with drafting reduced from 90 to 35 minutes, would save about 18.3 hours per month. This is a scenario, not a measured result. Measure drafting time, review effort, rework, factual errors and user satisfaction during a pilot.')
