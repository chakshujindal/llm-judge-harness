UV ?= uv
LJH = $(UV) run --quiet ljh
FLAGS = --no-cache --quiet
ANSWERS = examples/fake/answers.yaml
DEMO = runs/demo

.PHONY: install lint format test demo clean

install:
	$(UV) sync

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

test:
	$(UV) run pytest --cov --cov-report=term-missing

# No API key needed: every model here is a fake. Prints the JSON-extraction summary,
# then shows the CI gate by comparing the factual-QA run with a changed version of it.
demo:
	@rm -rf $(DEMO)
	@echo "=== ljh run: json-extraction (schema + LLM judge, 3 repeats) ==="
	@$(LJH) run examples/json_extraction.yaml -m fake:strong -m fake:weak --judge fake:judge \
		--fake-answers $(ANSWERS) $(FLAGS) --out $(DEMO)/json-extraction
	@$(LJH) run examples/factual_qa.yaml -m fake:echo -m fake:strong -m fake:weak \
		--fake-answers $(ANSWERS) $(FLAGS) --out $(DEMO)/factual-qa > /dev/null
	@$(LJH) run examples/factual_qa.yaml -m fake:echo -m fake:strong -m fake:weak \
		--fake-answers examples/fake/answers_v2.yaml $(FLAGS) --out $(DEMO)/factual-qa-v2 > /dev/null
	@echo
	@echo "=== ljh compare: factual-qa baseline vs changed models (exits 1 on any regression) ==="
	@$(LJH) compare $(DEMO)/factual-qa $(DEMO)/factual-qa-v2 \
		|| echo "(compare exited $$?: a regression was found, so a CI job would fail here)"
	@echo
	@echo "HTML reports: $(DEMO)/json-extraction/report.html, $(DEMO)/factual-qa/report.html"

clean:
	rm -rf runs .ljh_cache.sqlite .coverage htmlcov .pytest_cache .ruff_cache
